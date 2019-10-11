package com.databricks.labs.automl.pipeline

import java.util.UUID

import com.databricks.labs.automl.exceptions.{DateFeatureConversionException, FeatureConversionException, TimeFeatureConversionException}
import com.databricks.labs.automl.params.{GroupedModelReturn, MainConfig}
import com.databricks.labs.automl.sanitize.Scaler
import com.databricks.labs.automl.utils.{AutoMlPipelineUtils, SchemaUtils}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Model, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.DataFrame
import com.databricks.labs.automl.pipeline.PipelineVars._
import org.apache.log4j.Logger

/**
  * @author Jas Bali
  * This singleton encapsulates generation of feature engineering pipeline as well as inference pipeline, given
  * [[MainConfig]] and input [[DataFrame]]
  */

import scala.collection.mutable.ArrayBuffer

final case class VectorizationOutput(pipelineModel: PipelineModel, vectorizedCols: Array[String])

final case class FeatureEngineeringOutput(pipelineModel: PipelineModel,
                                          originalDfViewName: String, decidedModel: String,
                                          transformedForTrainingDf: DataFrame)

object FeatureEngineeringPipelineContext {

  @transient lazy private val logger: Logger = Logger.getLogger(this.getClass)

  //TODO (Jas): verbose true, only works for only feature engineering pipeline, for full predict pipeline this needs to be update.
  def generatePipelineModel(originalInputDataset: DataFrame,
                            mainConfig: MainConfig,
                            verbose: Boolean = false): FeatureEngineeringOutput = {

    val originalDfTempTableName = Identifiable.randomUID("zipWithId")

    val removeColumns = new ArrayBuffer[String]

    // First Transformation: Select required columns, convert date/time features and apply cardinality limit
    val initialPipelineModel = selectFeaturesConvertTypesAndApplyCardLimit(originalInputDataset, mainConfig, originalDfTempTableName)
    val initialTransformationDf = initialPipelineModel.transform(originalInputDataset)

    // Second Transformation: Apply string indexers, apply vector assembler, drop unnecessary columns
    val secondTransformation = applyStngIndxVectAssembler(
      initialTransformationDf,
      mainConfig,
      Array(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
    )
    val vectorizedColumns = secondTransformation.vectorizedCols
    removeColumns ++= vectorizedColumns
    val secondTransformationPipelineModel = secondTransformation.pipelineModel
    val secondTransformationDf = secondTransformationPipelineModel.transform(initialTransformationDf)

    val stages = new ArrayBuffer[PipelineStage]()
    // Fill with Na
    getAndAddStage(stages, fillNaStage(mainConfig))

    // Apply Outlier Filtering
    getAndAddStage(stages, outlierFilterStage(mainConfig))

    // Apply Vector Assembler
    getAndAddStage(stages, vectorAssemblerStage(mainConfig, vectorizedColumns))

    // Apply Variance filter
    getAndAddStage(stages, varianceFilterStage(mainConfig))

    // Apply Covariance Filtering
    getAndAddStage(stages, covarianceFilteringStage(mainConfig))

    // Apply Pearson Filtering
    getAndAddStage(stages, pearsonFilteringStage(mainConfig))

    // Third Transformation
    val thirdPipelineModel = new Pipeline().setStages(stages.toArray).fit(secondTransformationDf)
    val thirdTransformationDf = thirdPipelineModel.transform(secondTransformationDf)
    val oheInputCols = thirdTransformationDf
      .columns
      .filter(item => item.endsWith(PipelineEnums.SI_SUFFIX.value))
      .filterNot(item => (mainConfig.labelCol+PipelineEnums.SI_SUFFIX.value).equals(item))

    // Ksampler stages
    val ksampleStages = ksamplerStages(mainConfig)
    var ksampledDf = thirdTransformationDf
    if(ksampleStages.isDefined) {
      val ksamplerPipelineModel = new Pipeline().setStages(ksampleStages.get).fit(thirdTransformationDf)
      ksampledDf = ksamplerPipelineModel.transform(thirdTransformationDf)
    }

    val lastStages = new ArrayBuffer[PipelineStage]()
    // Roundup OHE input Cols
    getAndAddStage(lastStages, Some(new RoundUpDoubleTransformer().setInputCols(oheInputCols)))
    // Apply OneHotEncoding Options
    getAndAddStage(lastStages, oneHotEncodingStage(mainConfig, oheInputCols))
    getAndAddStage(lastStages, dropColumns(Array(mainConfig.featuresCol), mainConfig))
    // Execute Vector Assembler Again
    getAndAddStage(
      lastStages,
      vectorAssemblerStage(
        mainConfig, oheInputCols.map(SchemaUtils.generateOneHotEncodedColumn)
        ++ vectorizedColumns.filterNot(_.endsWith(PipelineEnums.SI_SUFFIX.value))))

    // Apply Scaler option
    val renamedFeatureCol = mainConfig.featuresCol + PipelineEnums.FEATURE_NAME_TEMP_SUFFIX.value
    getAndAddStage(lastStages, renameTransformerStage(mainConfig.featuresCol, renamedFeatureCol, mainConfig))
    getAndAddStage(lastStages, scalerStage(mainConfig))
    getAndAddStage(lastStages, dropColumns(Array(renamedFeatureCol), mainConfig))

    // Drop Unnecessary columns - output of feature engineering stage should only contain automl_internal_id, label, features and synthetic from ksampler
    removeColumns ++= oheInputCols.map(SchemaUtils.generateOneHotEncodedColumn)
    if(!verbose) {
      getAndAddStage(lastStages, dropColumns(removeColumns.toArray, mainConfig))
    }
    // final transformation
    val fourthPipelineModel = new Pipeline().setStages(lastStages.toArray).fit(ksampledDf)
    val fourthTransformationDf = fourthPipelineModel.transform(ksampledDf)

    //Extract Decided model from DataSanitizer stage
    val dataSanitizerStage = thirdPipelineModel.stages.find(item => item.isInstanceOf[DataSanitizerTransformer]).get

    FeatureEngineeringOutput(
      mergePipelineModels(ArrayBuffer(initialPipelineModel, secondTransformationPipelineModel, thirdPipelineModel, fourthPipelineModel)),
      originalDfTempTableName,
      dataSanitizerStage.getOrDefault(dataSanitizerStage.getParam("decideModel")).asInstanceOf[String],
      fourthTransformationDf
    )
  }

  def buildFullPredictPipeline(featureEngOutput: FeatureEngineeringOutput,
                               modelReport: Array[GroupedModelReturn],
                               mainConfiguration: MainConfig,
                               originalDf: DataFrame): PipelineModel = {
    val pipelineModelStages = new ArrayBuffer[PipelineModel]()
    //Build Pipeline here
    // get Feature eng. pipeline model
    pipelineModelStages += featureEngOutput.pipelineModel

    val bestModel = getBestModel(modelReport, mainConfiguration.scoringOptimizationStrategy)
    val mlPipelineModel = SparkUtil.createPipelineModel(Array(bestModel.model.asInstanceOf[Model[_]]))

    pipelineModelStages += mlPipelineModel
    val pipelinewithMlModel = FeatureEngineeringPipelineContext.mergePipelineModels(pipelineModelStages)
    val pipelinewithMlModelDf = mlPipelineModel.transform(featureEngOutput.transformedForTrainingDf)

    // Add Index To String Stage
    val pipelineModelWithLabelSi = addLabelIndexToString(
      pipelinewithMlModel,
      pipelinewithMlModelDf,
      mainConfiguration)
    val pipelineModelWithLabelSiDf = pipelineModelWithLabelSi.transform(originalDf)

    val finalPipelineModel = addUserReturnViewStage(
      pipelineModelWithLabelSi,
      mainConfiguration,
      pipelineModelWithLabelSiDf,
      featureEngOutput.originalDfViewName)

    // Removes train-only stages, if present, such as OutlierTransformer and SyntheticDataTransformer
    lintPipelineStages(finalPipelineModel)
  }

  private def lintPipelineStages(pipelineModel: PipelineModel): PipelineModel = {
    val nonTrainingStages = pipelineModel.stages.filterNot(_.isInstanceOf[IsTrainingStage])
    logger.debug(
      s"""Removed following training stages from inference-only pipeline ${nonTrainingStages.map(_.uid).mkString(", ")}""")
    SparkUtil.createPipelineModel(
      Identifiable.randomUID("final_linted_infer_pipeline"),
      nonTrainingStages)
  }

  private def getBestModel(runData: Array[GroupedModelReturn],
                           optimizationStrategy: String): GroupedModelReturn = {
    optimizationStrategy match {
      case "minimize" => runData.sortWith(_.score < _.score)(0)
      case _ => runData.sortWith(_.score > _.score)(0)
    }
  }

  private def addLabelIndexToString(pipelineModel: PipelineModel,
                        dataFrame: DataFrame,
                        mainConfig: MainConfig): PipelineModel = {
     if(SchemaUtils.isLabelRefactorNeeded(dataFrame.schema, mainConfig.labelCol)
       ||
       PipelineStateCache
         .getFromPipelineByIdAndKey(
             mainConfig.pipelineId,
             PIPELINE_LABEL_REFACTOR_NEEDED_KEY.key)
         .asInstanceOf[Boolean]
       ) {
       //Find the last string indexer by reversing the pipeline mode stages
       val stringIndexerLabels =
         pipelineModel
         .stages
         .find(_.uid.startsWith(PipelineEnums.LABEL_STRING_INDEXER_STAGE_NAME.value))
         .get
         .asInstanceOf[StringIndexerModel]
         .labels

       val labelRefactorPipelineModel =  new Pipeline()
         .setStages(Array(
           new IndexToString()
           .setInputCol("prediction")
           .setOutputCol("prediction_stng")
           .setLabels(stringIndexerLabels),
           new DropColumnsTransformer()
             .setInputCols(Array("prediction")),
           new ColumnNameTransformer()
             .setInputColumns(Array("prediction_stng"))
             .setOutputColumns(Array("prediction"))
         ))
         .fit(dataFrame)
       labelRefactorPipelineModel.transform(dataFrame)

       return mergePipelineModels(ArrayBuffer(pipelineModel, labelRefactorPipelineModel))
     }
     pipelineModel
  }

  private def getInputFeautureCols(inputDataFrame: DataFrame,
                                   mainConfig: MainConfig): Array[String] = {
    inputDataFrame.columns
      .filterNot(mainConfig.fieldsToIgnoreInVector.contains)
      .filterNot(Array(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL).contains)
      .filterNot(Array(mainConfig.labelCol).contains)
  }

  def addUserReturnViewStage(pipelineModel: PipelineModel,
                             mainConfig: MainConfig,
                             dataFrame: DataFrame,
                             originalDfTempTableName: String): PipelineModel = {
    // Generate output dataset
    val inputFeatures = getInputFeautureCols(dataFrame.sqlContext.sql(s"select * from $originalDfTempTableName"), mainConfig)

    val userViewPipelineModel = new Pipeline().setStages(
      Array(new AutoMlOutputDatasetTransformer()
        .setTempViewOriginalDatasetName(originalDfTempTableName)
        .setLabelColumn(mainConfig.labelCol)
        .setFeatureColumns(inputFeatures)))
    .fit(dataFrame)

    userViewPipelineModel.transform(dataFrame)

    mergePipelineModels(ArrayBuffer(pipelineModel, userViewPipelineModel))
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
    val inputFeatures = getInputFeautureCols(dataFrame, mainConfig)

    val zipRegisterTempTransformer = new ZipRegisterTempTransformer()
      .setTempViewOriginalDatasetName(originalDfTempTableName)
      .setLabelColumn(mainConfig.labelCol)
      .setFeatureColumns(inputFeatures)
      .setDebugEnabled(mainConfig.pipelineDebugFlag)

    val mlFlowLoggingValidationStageTransformer = new MlFlowLoggingValidationStageTransformer()
      .setMlFlowAPIToken(mainConfig.mlFlowConfig.mlFlowAPIToken)
      .setMlFlowTrackingURI(mainConfig.mlFlowConfig.mlFlowTrackingURI)
      .setMlFlowExperimentName(mainConfig.mlFlowConfig.mlFlowExperimentName)
      .setMlFlowLoggingFlag(mainConfig.mlFlowLoggingFlag)
      .setDebugEnabled(mainConfig.pipelineDebugFlag)

    val cardinalityLimitColumnPrunerTransformer = new CardinalityLimitColumnPrunerTransformer()
      .setLabelColumn(mainConfig.labelCol)
      .setCardinalityLimit(mainConfig.fillConfig.cardinalityLimit)
      .setCardinalityCheckMode(mainConfig.fillConfig.cardinalityCheckMode)
      .setCardinalityPrecision(mainConfig.fillConfig.cardinalityPrecision)
      .setCardinalityType(mainConfig.fillConfig.cardinalityType)
      .setDebugEnabled(mainConfig.pipelineDebugFlag)

    val dateFieldTransformer = new DateFieldTransformer()
      .setLabelColumn(mainConfig.labelCol)
      .setMode(mainConfig.dateTimeConversionType)
      .setDebugEnabled(mainConfig.pipelineDebugFlag)

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
    * @param ignoreCols
    * @return
    */
  private def applyStngIndxVectAssembler(dataFrame: DataFrame,
                                         mainConfig: MainConfig,
                                         ignoreCols: Array[String]): VectorizationOutput = {
    val fields = SchemaUtils.extractTypes(dataFrame, mainConfig.labelCol)
    val stringFields = fields._2.filterNot(ignoreCols.contains).filterNot(item => item.equals(mainConfig.labelCol))
    val vectorizableFields = fields._1.toArray.filterNot(ignoreCols.contains)
    val dateFields = fields._3.toArray.filterNot(ignoreCols.contains)
    val timeFields = fields._4.toArray.filterNot(ignoreCols.contains)

    //Validate date and time fields are empty at this point
    validateDateAndTimeFeatures(dateFields, timeFields)

    val stages = new ArrayBuffer[PipelineStage]

    // Label refactor
    if(SchemaUtils.isLabelRefactorNeeded(dataFrame.schema, mainConfig.labelCol)) {
      getAndAddStage(
        stages,
        Some(new StringIndexer(PipelineEnums.LABEL_STRING_INDEXER_STAGE_NAME.value + Identifiable.randomUID("strIdx"))
          .setInputCol(mainConfig.labelCol)
          .setOutputCol(mainConfig.labelCol+PipelineEnums.SI_SUFFIX.value)
          .setHandleInvalid("keep")))
      getAndAddStage(stages, dropColumns(Array(mainConfig.labelCol), mainConfig))
      getAndAddStage(stages,
        renameTransformerStage(
          mainConfig.labelCol+PipelineEnums.SI_SUFFIX.value,
          mainConfig.labelCol,
          mainConfig))
      // Register label refactor needed var for this pipeline context
      // LabelRefactor needed
      addToPipelineCacheInternal(mainConfig, refactorNeeded = true)
    } else {
      // Register label refactor needed var for this pipeline context
      //Label refactor not required
      addToPipelineCacheInternal(mainConfig, refactorNeeded= false)
    }
    stringFields.foreach(columnName => {
      stages += new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(SchemaUtils.generateStringIndexedColumn(columnName))
        .setHandleInvalid("keep")
    }
    )
    stages += new DropColumnsTransformer().setInputCols(stringFields.toArray).setDebugEnabled(mainConfig.pipelineDebugFlag)

    val featureAssemblerInputCols: Array[String] = stringFields
      .map(item => SchemaUtils.generateStringIndexedColumn(item))
      .toArray[String] ++ vectorizableFields

    VectorizationOutput(new Pipeline().setStages(stages.toArray).fit(dataFrame), featureAssemblerInputCols)
  }

  private def addToPipelineCacheInternal(mainConfig: MainConfig, refactorNeeded: Boolean): Unit = {
    PipelineStateCache
      .addToPipelineCache(
        mainConfig.pipelineId,
        PipelineVars.PIPELINE_LABEL_REFACTOR_NEEDED_KEY.key, refactorNeeded)
  }

  private def vectorAssemblerStage(mainConfig: MainConfig,
                                   featureAssemblerInputCols: Array[String]): Option[PipelineStage] = {
    Some(new VectorAssembler()
      .setInputCols(featureAssemblerInputCols)
      .setOutputCol(mainConfig.featuresCol))
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
       .setDebugEnabled(mainConfig.pipelineDebugFlag)

    Some(dataSanitizerTransformer)
  }

  private def varianceFilterStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.varianceFilterFlag) {
      val varianceFilterTransformer = new VarianceFilterTransformer()
        .setLabelColumn(mainConfig.labelCol)
        .setFeatureCol(mainConfig.featuresCol)
        .setDebugEnabled(mainConfig.pipelineDebugFlag)
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
        .setLabelColumn(mainConfig.labelCol)
        .setDebugEnabled(mainConfig.pipelineDebugFlag)
      return Some(outlierFilterTransformer)
    }
    None
  }

  private def covarianceFilteringStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.covarianceFilteringFlag) {
      val covarianceFilterTransformer = new CovarianceFilterTransformer()
        .setLabelColumn(mainConfig.labelCol)
        .setCorrelationCutoffLow(mainConfig.covarianceConfig.correlationCutoffHigh)
        .setCorrelationCutoffHigh(mainConfig.covarianceConfig.correlationCutoffLow)
        .setDebugEnabled(mainConfig.pipelineDebugFlag)
      return Some(covarianceFilterTransformer)
    }
    None
  }

  private def pearsonFilteringStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.pearsonFilteringFlag) {
      val pearsonFilterTransformer = new PearsonFilterTransformer()
        .setLabelColumn(mainConfig.labelCol)
        .setFeatureCol(mainConfig.featuresCol)
        .setAutoFilterNTile(mainConfig.pearsonConfig.autoFilterNTile)
        .setFilterDirection(mainConfig.pearsonConfig.filterDirection)
        .setFilterManualValue(mainConfig.pearsonConfig.filterManualValue)
        .setFilterMode(mainConfig.pearsonConfig.filterMode)
        .setFilterStatistic(mainConfig.pearsonConfig.filterStatistic)
        .setDebugEnabled(mainConfig.pipelineDebugFlag)
      return Some(pearsonFilterTransformer)
    }
    None
  }

  private def oneHotEncodingStage(mainConfig: MainConfig, stngIndxCols: Array[String]): Option[PipelineStage] = {
    if(mainConfig.oneHotEncodeFlag) {
      return Some(new OneHotEncoderEstimator()
        .setInputCols(stngIndxCols)
        .setOutputCols(stngIndxCols.map(item => SchemaUtils.generateOneHotEncodedColumn(item)))
        .setHandleInvalid("keep"))
    }
    None
  }

  private def scalerStage(mainConfig: MainConfig): Option[PipelineStage] = {
    if(mainConfig.scalingFlag) {
      return Some(new Scaler()
        .setFeaturesCol(mainConfig.featuresCol)
        .setScalerType(mainConfig.scalingConfig.scalerType)
        .setScalerMin(mainConfig.scalingConfig.scalerMin)
        .setScalerMax(mainConfig.scalingConfig.scalerMax)
        .setStandardScalerMeanMode(
          mainConfig.scalingConfig.standardScalerMeanFlag
        )
        .setStandardScalerStdDevMode(
          mainConfig.scalingConfig.standardScalerStdDevFlag
        )
        .setPNorm(mainConfig.scalingConfig.pNorm)
        .scaleFeaturesForPipeline())
    }
    None
  }

  private def ksamplerStages(mainConfig: MainConfig): Option[Array[_ <: PipelineStage]] = {
    if (mainConfig.geneticConfig.trainSplitMethod == "kSample") {
      val arrayBuffer = new ArrayBuffer[PipelineStage]()
      // Ksampler stage
      arrayBuffer += new SyntheticFeatureGenTransformer()
        .setFeatureCol(mainConfig.featuresCol)
        .setLabelColumn(mainConfig.labelCol)
        .setSyntheticCol(mainConfig.geneticConfig.kSampleConfig.syntheticCol)
        .setKGroups(mainConfig.geneticConfig.kSampleConfig.kGroups)
        .setKMeansMaxIter(mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter)
        .setKMeansTolerance(mainConfig.geneticConfig.kSampleConfig.kMeansTolerance)
        .setKMeansDistanceMeasurement(mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement)
        .setKMeansSeed(mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
        .setKMeansPredictionCol(mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol)
        .setLshHashTables(mainConfig.geneticConfig.kSampleConfig.lshHashTables)
        .setLshSeed(mainConfig.geneticConfig.kSampleConfig.lshSeed)
        .setLshOutputCol(mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
        .setQuorumCount(mainConfig.geneticConfig.kSampleConfig.quorumCount)
        .setMinimumVectorCountToMutate(mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate)
        .setVectorMutationMethod(mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod)
        .setMutationMode(mainConfig.geneticConfig.kSampleConfig.mutationMode)
        .setMutationValue(mainConfig.geneticConfig.kSampleConfig.mutationValue)
        .setLabelBalanceMode(mainConfig.geneticConfig.kSampleConfig.labelBalanceMode)
        .setCardinalityThreshold(mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold)
        .setNumericRatio(mainConfig.geneticConfig.kSampleConfig.numericRatio)
        .setNumericTarget(mainConfig.geneticConfig.kSampleConfig.numericTarget)
        .setDebugEnabled(mainConfig.pipelineDebugFlag)

      //Repartition after Ksampler stage
      arrayBuffer += new RepartitionTransformer()
        .setPartitionScaleFactor(mainConfig.geneticConfig.kSampleConfig.outputDfRepartitionScaleFactor)

      // Register temp table stage for registering non-synthetic dataset later needed for the union with synthetic dataset
      val nonSyntheticFeatureGenTmpTable = Identifiable.randomUID("nonSyntheticFeatureGenTransformer_")
      getAndAddStage(arrayBuffer,
        Some(new RegisterTempTableTransformer()
          .setTempTableName(nonSyntheticFeatureGenTmpTable)
          .setStatement(s"select * from __THIS__ where !${mainConfig.geneticConfig.kSampleConfig.syntheticCol}")
          .setDebugEnabled(mainConfig.pipelineDebugFlag)))

      // Get synthetic dataset
      getAndAddStage(arrayBuffer, Some(new SQLWrapperTransformer()
        .setStatement(s"select * from __THIS__ where ${mainConfig.geneticConfig.kSampleConfig.syntheticCol}")
        .setDebugEnabled(mainConfig.pipelineDebugFlag)))
      // If scaling is used, make sure that the synthetic data has the same scaling.
      if (mainConfig.scalingFlag) {
        val renamedFeatureCol = mainConfig.featuresCol + PipelineEnums.FEATURE_NAME_TEMP_SUFFIX.value
        getAndAddStage(arrayBuffer, renameTransformerStage(mainConfig.featuresCol, renamedFeatureCol, mainConfig))
        getAndAddStage(arrayBuffer, scalerStage(mainConfig))
        getAndAddStage(arrayBuffer, dropColumns(Array(renamedFeatureCol), mainConfig))
        arrayBuffer += new DatasetsUnionTransformer().setUnionDatasetName(nonSyntheticFeatureGenTmpTable)
        arrayBuffer += new DropTempTableTransformer().setTempTableName(nonSyntheticFeatureGenTmpTable)
      }

      return Some(arrayBuffer.toArray)
    }
    None
  }

  private def renameTransformerStage(oldLabelName: String,
                                     newLabelName: String,
                                     mainConfig: MainConfig): Option[PipelineStage] = {
    Some(new ColumnNameTransformer()
      .setInputColumns(Array(oldLabelName))
      .setOutputColumns(Array(newLabelName))
      .setDebugEnabled(mainConfig.pipelineDebugFlag))
  }

  private def dropColumns(colNames: Array[String], mainConfig: MainConfig): Option[PipelineStage] = {
    Some(new DropColumnsTransformer().setInputCols(colNames).setDebugEnabled(mainConfig.pipelineDebugFlag))
  }

  private def mergePipelineModels(pipelineModels: ArrayBuffer[PipelineModel]): PipelineModel = {
    SparkUtil.createPipelineModel(
      "final_ml_pipeline_"+UUID.randomUUID().toString,
      pipelineModels.flatMap(item => item.stages).toArray)
  }

  private def getAndAddStage[T](stages: ArrayBuffer[PipelineStage], value: Option[_ <: PipelineStage]): Unit = {
    if(value.isDefined) {
      stages += value.get
    }
  }

  private def getAndAddStages[T](stages: ArrayBuffer[PipelineStage], value: Option[Array[_ <: PipelineStage]]): Unit = {
    if(value.isDefined) {
      stages ++= value.get
    }
  }
}

