package com.databricks.labs.automl.pipeline

import java.util.UUID

import com.databricks.labs.automl.params.MainConfig
import com.databricks.labs.automl.utils.SchemaUtils
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

object FeatureEngineeringPipelineContext {

  def generatePipelineModel(originalInputDataset: DataFrame,
                       mainConfig: MainConfig): PipelineModel = {

    val originalDfTempTableName = UUID.randomUUID().toString

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
    stages += getStage(fillNaStage(mainConfig))

    // Fourth Transformation: Apply Variance filter
    stages += getStage(varianceFilterStage(mainConfig))

    // Fifth Transformation: Apply Outlier Filtering
    stages += getStage(outlierFilterStage(mainConfig))

    // Sixth Transformation: Apply Covariance Filtering
    stages += getStage(covarianceFilteringStage(mainConfig))

    //Seventh Tranformation: Apply Pearson Filtering
    stages += getStage(pearsonFilteringStage(mainConfig))

    // Apply OneHotEncoding Options
    stages += getStage(oneHotEncodingStage(mainConfig))

    // Apply Scaler option
    stages += getStage(scalerStage(mainConfig))


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
    val fieldsToBeIndexed = SchemaUtils.extractTypes(dataFrame, mainConfig.labelCol)._2
    val stages = new ArrayBuffer[PipelineStage]
    fieldsToBeIndexed.foreach(columnName => {
      stages += new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(SchemaUtils.generateStringIndexedColumn(columnName))
        .setHandleInvalid("keep")
    }
    )
    stages += new DropColumnsTransformer().setInputCols(fieldsToBeIndexed.toArray)

    val featureAssemblerInputCols = fieldsToBeIndexed.map(item => SchemaUtils.generateStringIndexedColumn(item)).toArray

    stages += new VectorAssembler()
      .setInputCols(featureAssemblerInputCols)
        .setOutputCol(mainConfig.featuresCol)

    stages += new DropColumnsTransformer().setInputCols(featureAssemblerInputCols)

    new Pipeline().setStages(stages.toArray).fit(dataFrame)
  }

  private def fillNaStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.naFillFlag) {
     val dataSanitizerTransformer = new DataSanitizerTransformer()
        .setLabelColumn(mainConfig.labelCol)
        .setFeatureCol(mainConfig.featuresCol)
        .setModelSelectionDistinctThreshold(
          mainConfig.fillConfig.modelSelectionDistinctThreshold
        )
        .setNumericFillStat(mainConfig.fillConfig.numericFillStat)
        .setCharacterFillStat(mainConfig.fillConfig.characterFillStat)
        .setParallelism(mainConfig.geneticConfig.parallelism)

      return Some(dataSanitizerTransformer)
    }
    None
  }

  private def varianceFilterStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.varianceFilterFlag) {
      val varianceFilterTransformer = new VarianceFilterTransformer()
      return Some(varianceFilterTransformer)
    }
    None
  }

  private def outlierFilterStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.outlierFilterFlag) {
      val outlierFilterTransformer = new OutlierFilterTransformer()
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

  //TODO: find elegant way to merge PipelineModels
  private def mergePipelineModels(pipelineModels: ArrayBuffer[PipelineModel]): PipelineModel = {
    pipelineModels(0)
//    new PipelineModel(pipelineModels.flatMap(item => item.stages))
  }

  def getStage[T](value: Option[T]): T = {
    value.filterNot(_.isInstanceOf[T]).get
  }
}

