package com.databricks.labs.automl.pipeline

import java.util.UUID

import com.databricks.labs.automl.exceptions.{DateFeatureConversionException, FeatureConversionException, StringFeatureConversionException, TimeFeatureConversionException}
import com.databricks.labs.automl.params.MainConfig
import com.databricks.labs.automl.utils.SchemaUtils
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

object FeatureEngineeringPipelineContext {

  def generatePipelineModel(originalInputDataset: DataFrame,
                       mainConfig: MainConfig): PipelineModel = {

    val originalDfTempTableName = Identifiable.randomUID("zipWithId")

    val initialpipelineModels = new ArrayBuffer[PipelineModel]

    // First Transformation: Select required columns, convert date/time features and apply cardinality limit
    val initialPipelineModel = selectFeaturesConvertTypesAndApplyCardLimit(originalInputDataset, mainConfig, originalDfTempTableName)
    val initialTransformationDf = initialPipelineModel.transform(originalInputDataset)
    initialpipelineModels += initialPipelineModel

    // Second Transformation: Apply string indexers, apply vector assembler, drop unnecessary columns
    val secondTransformationPipelineModel = applyStngIndxVectAssembler(initialTransformationDf, mainConfig, originalDfTempTableName)
    val secondTransformationDf = secondTransformationPipelineModel.transform(initialTransformationDf)
    initialpipelineModels += secondTransformationPipelineModel

    val stages = new ArrayBuffer[PipelineStage]()
    // Third Transformation: Fill with Na
    getAndAddStage(stages, fillNaStage(mainConfig))

    // Fourth Transformation: Apply Variance filter
    getAndAddStage(stages, varianceFilterStage(mainConfig))

    // Fifth Transformation: Apply Outlier Filtering
    getAndAddStage(stages, outlierFilterStage(mainConfig))

    // Sixth Transformation: Apply Covariance Filtering
    getAndAddStage(stages, covarianceFilteringStage(mainConfig))

    //Seventh Tranformation: Apply Pearson Filtering
    getAndAddStage(stages, pearsonFilteringStage(mainConfig))

    // Apply OneHotEncoding Options
    getAndAddStage(stages, oneHotEncodingStage(mainConfig))

    // Apply Scaler option
    getAndAddStage(stages, scalerStage(mainConfig))


    mergePipelineModels(initialpipelineModels += new Pipeline().setStages(stages.toArray).fit(secondTransformationDf))

  }

  /**
    * Select feature columns, converting date/time features and applying cardinality limit stages
    *
    * @param dataFrame
    * @param mainConfig
    * @param originalDfTempTableName
    * @return
    */
  private def selectFeaturesConvertTypesAndApplyCardLimit(dataFrame: DataFrame,
                                  mainConfig: MainConfig,
                                  originalDfTempTableName: String): PipelineModel = {
    // Stage to select only those columns that are needed in the downstream stages
    // also creates a temp view of the original dataset which will then be used by the last stage
    // to return user table
    val zipRegisterTempTransformer = new ZipRegisterTempTransformer()
      .setTempViewOriginalDatasetName(originalDfTempTableName)
      .setLabelColumn(mainConfig.labelCol)
      .setFeatureColumns(mainConfig.inputFeatures)

    val mlFlowLoggingValidationStageTransformer = new MlFlowLoggingValidationStageTransformer()
      .setMlFlowAPIToken(mainConfig.mlFlowConfig.mlFlowAPIToken)
      .setMlFlowTrackingURI(mainConfig.mlFlowConfig.mlFlowTrackingURI)
      .setMlFlowExperimentName(mainConfig.mlFlowConfig.mlFlowExperimentName)
      .setMlFlowLoggingFlag(mainConfig.mlFlowLoggingFlag)

    val cardinalityLimitColumnPrunerTransformer = new CardinalityLimitColumnPrunerTransformer()
      .setLabelColumn(mainConfig.labelCol)
      .setCardinalityLimit(500)

    val dateFieldTransformer = new DateFieldTransformer()
      .setLabelColumn(mainConfig.labelCol)
      .setMode(mainConfig.dateTimeConversionType)

    new Pipeline().setStages(Array(
        zipRegisterTempTransformer,
        mlFlowLoggingValidationStageTransformer,
        cardinalityLimitColumnPrunerTransformer,
        dateFieldTransformer)
    ).fit(dataFrame)
  }

  /**
    * Apply string indexers, apply vector assembler, drop unnecessary columns
    * @param dataFrame
    * @param mainConfig
    * @param originalDfTempTableName
    * @return
    */
  private def applyStngIndxVectAssembler(dataFrame: DataFrame,
                                  mainConfig: MainConfig,
                                  originalDfTempTableName: String): PipelineModel = {
    val fields = SchemaUtils.extractTypes(dataFrame, mainConfig.labelCol)
    val stringFields = fields._2
    val vectorizableFields = fields._1.toArray
    val dateFields = fields._3.toArray
    val timeFields = fields._4.toArray

    //Validate date and time fields are empty at this point
    validateDateAndTimeFeatures(dateFields, timeFields)

    val stages = new ArrayBuffer[PipelineStage]
    stringFields.foreach(columnName => {
      stages += new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(SchemaUtils.generateStringIndexedColumn(columnName))
        .setHandleInvalid("keep")
    }
    )
    stages += new DropColumnsTransformer().setInputCols(stringFields.toArray)

    val featureAssemblerInputCols: Array[String] = stringFields
      .map(item => SchemaUtils.generateStringIndexedColumn(item))
      .toArray[String] ++ vectorizableFields

    stages += new VectorAssembler()
      .setInputCols(featureAssemblerInputCols)
        .setOutputCol(mainConfig.featuresCol)

    stages += new DropColumnsTransformer().setInputCols(featureAssemblerInputCols)

    new Pipeline().setStages(stages.toArray).fit(dataFrame)
  }

  private def validateDateAndTimeFeatures(dateFields: Array[String],
                             timeFields: Array[String]): Unit = {
    throwFieldConversionException(dateFields, classOf[DateFeatureConversionException])
    throwFieldConversionException(timeFields, classOf[TimeFeatureConversionException])
  }

  private def throwFieldConversionException(fields: Array[_ <: String],
                                            clazz: Class[_ <:FeatureConversionException]): Unit = {
    if(SchemaUtils.isNotEmpty(fields)) {
      throw clazz.getConstructor(classOf[Array[String]]).newInstance(fields)
    }
  }

  private def fillNaStage(mainConfig: MainConfig): Option[PipelineStage] = {
     val dataSanitizerTransformer = new DataSanitizerTransformer()
        .setLabelColumn(mainConfig.labelCol)
        .setFeatureCol(mainConfig.featuresCol)
        .setModelSelectionDistinctThreshold(
          mainConfig.fillConfig.modelSelectionDistinctThreshold
        )
        .setNumericFillStat(mainConfig.fillConfig.numericFillStat)
        .setCharacterFillStat(mainConfig.fillConfig.characterFillStat)
        .setParallelism(mainConfig.geneticConfig.parallelism)
       .setCategoricalNAFillMap(mainConfig.fillConfig.categoricalNAFillMap)
       .setNumericNAFillMap(mainConfig.fillConfig.numericNAFillMap.asInstanceOf[Map[String, Double]])
       .setFillMode(mainConfig.fillConfig.naFillMode)
       .setFilterPrecision(mainConfig.fillConfig.filterPrecision)
       .setNumericNABlanketFill(mainConfig.fillConfig.numericNABlanketFillValue)
       .setCharacterNABlanketFill(mainConfig.fillConfig.characterNABlanketFillValue)
       .setNaFillFlag(mainConfig.naFillFlag)
    Some(dataSanitizerTransformer)
  }

  private def varianceFilterStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.varianceFilterFlag) {
      val varianceFilterTransformer = new VarianceFilterTransformer()
        .setLabelColumn(mainConfig.labelCol)
        .setFeatureCol(mainConfig.featuresCol)
      return Some(varianceFilterTransformer)
    }
    None
  }

  private def outlierFilterStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.outlierFilterFlag) {
      val outlierFilterTransformer = new OutlierFilterTransformer()
        .setFilterBounds(mainConfig.outlierConfig.filterBounds)
        .setLowerFilterNTile(mainConfig.outlierConfig.lowerFilterNTile)
        .setUpperFilterNTile(mainConfig.outlierConfig.upperFilterNTile)
        .setFilterPrecision(mainConfig.outlierConfig.filterPrecision)
        .setContinuousDataThreshold(mainConfig.outlierConfig.continuousDataThreshold)
        .setParallelism(mainConfig.geneticConfig.parallelism)
        .setFieldsToIgnore(Array.empty)
      return Some(outlierFilterTransformer)
    }
    None
  }

  private def covarianceFilteringStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.covarianceFilteringFlag) {
      val covarianceFilterTransformer = new CovarianceFilterTransformer()
      return Some(covarianceFilterTransformer)
    }
    None
  }

  private def pearsonFilteringStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.pearsonFilteringFlag) {
      val pearsonFilterTransformer = new PearsonFilterTransformer()
      return Some(pearsonFilterTransformer)
    }
    None
  }

  private def oneHotEncodingStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.oneHotEncodeFlag) {
      Some
    }
    None
  }

  private def scalerStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.scalingFlag) {
      Some
    }
    None
  }

  private def mergePipelineModels(pipelineModels: ArrayBuffer[PipelineModel]): PipelineModel = {
    SparkUtil.createPipelineModel(UUID.randomUUID().toString, pipelineModels.flatMap(item => item.stages).toArray)
//null
//    new PipelineModel(pipelineModels.flatMap(item => item.stages))
  }

  def getAndAddStage[T](stages: ArrayBuffer[PipelineStage], value: Option[_ <: PipelineStage]): Unit = {
    if(value.isDefined) {
      stages += value.get
    }
  }
}

