package com.databricks.labs.automl.executor

import com.databricks.labs.automl.feature.{
  FeatureInteraction,
  SyntheticFeatureGenerator
}
import com.databricks.labs.automl.feature.structures.{
  FeatureInteractionOutputPayload,
  InteractionPayloadExtract
}
import com.databricks.labs.automl.inference.{
  FeatureInteractionConfig,
  InferenceConfig,
  NaFillConfig
}
import com.databricks.labs.automl.params.{
  DataGeneration,
  DataPrepReturn,
  OutlierFilteringReturn
}
import com.databricks.labs.automl.pipeline.FeaturePipeline
import com.databricks.labs.automl.sanitize._
import com.databricks.labs.automl.utils.{
  AutomationTools,
  WorkspaceDirectoryValidation
}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

class DataPrep(df: DataFrame) extends AutomationConfig with AutomationTools {

  //    TODO: parallelism config for non genetic parallel control should be added
  private val logger: Logger = Logger.getLogger(this.getClass)

  private def logConfig(): Unit = {

    val configString = s"Configuration setting flags: \n NA Fill Flag: ${_mainConfig.naFillFlag.toString}" +
      s"\n Zero Variance Filter Flag: ${_mainConfig.varianceFilterFlag.toString}" +
      s"\n Outlier Filter Flag: ${_mainConfig.outlierFilterFlag.toString}" +
      s"\n Covariance Filter Flag: ${_mainConfig.covarianceFilteringFlag.toString}" +
      s"\n Pearson Filter Flag: ${_mainConfig.pearsonFilteringFlag.toString}" +
      s"\n OneHotEncoding Flag: ${_mainConfig.oneHotEncodeFlag.toString}" +
      s"\n Scaling Flag: ${_mainConfig.scalingFlag.toString}" +
      s"\n Feature Interaction Flag: ${_mainConfig.featureInteractionFlag.toString}" +
      s"\n Hyperspace Inference Flag: ${_mainConfig.geneticConfig.hyperSpaceInference.toString}" +
      s"\n First Generation Seed Mode: ${_mainConfig.geneticConfig.initialGenerationMode}" +
      s"\n MlFlow Logging Flag: ${_mainConfig.mlFlowLoggingFlag.toString}" +
      s"\n Early Stopping Flag: ${_mainConfig.autoStoppingFlag.toString}" +
      s"\n Data Prep Caching Flag: ${_mainConfig.dataPrepCachingFlag.toString}"
    println(
      configString + "\nFull Model Tuning Run Config: \n" + prettyPrintConfig(
        _mainConfig
      )
    )
    logger.log(Level.INFO, configString)

  }

  private def vectorPipeline(
    data: DataFrame,
    cardinalityFlag: Boolean
  ): (DataFrame, Array[String], Array[String]) = {

    // Creates the feature vector and returns the fields that go into the vector

    new FeaturePipeline(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .setDateTimeConversionType(_mainConfig.dateTimeConversionType)
      .setCardinalityCheck(cardinalityFlag)
      .setCardinalityCheckMode(_mainConfig.fillConfig.cardinalityCheckMode)
      .setCardinalityLimit(_mainConfig.fillConfig.cardinalityLimit)
      .setCardinalityPrecision(_mainConfig.fillConfig.cardinalityPrecision)
      .setCardinalityType(_mainConfig.fillConfig.cardinalityType)
      .makeFeaturePipeline(_mainConfig.fieldsToIgnoreInVector)

  }

  private def oneHotEncodeVector(
    data: DataFrame,
    featureColumns: Array[String],
    totalFields: Array[String]
  ): (DataFrame, Array[String], Array[String]) = {

    new FeaturePipeline(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .applyOneHotEncoding(featureColumns, totalFields)

  }

  private def fillNA(data: DataFrame): (DataFrame, NaFillConfig, String) = {

    // Output has no feature vector

    val naConfig = new DataSanitizer(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .setModelSelectionDistinctThreshold(
        _mainConfig.fillConfig.modelSelectionDistinctThreshold
      )
      .setNumericFillStat(_mainConfig.fillConfig.numericFillStat)
      .setCharacterFillStat(_mainConfig.fillConfig.characterFillStat)
      .setParallelism(_mainConfig.dataPrepParallelism)
      .setFieldsToIgnoreInVector(_mainConfig.fieldsToIgnoreInVector)
      .setFilterPrecision(_mainConfig.fillConfig.filterPrecision)
      .setCategoricalNAFillMap(_mainConfig.fillConfig.categoricalNAFillMap)
      .setNumericNAFillMap(_mainConfig.fillConfig.numericNAFillMap)
      .setCharacterNABlanketFillValue(
        _mainConfig.fillConfig.characterNABlanketFillValue
      )
      .setNumericNABlanketFillValue(
        _mainConfig.fillConfig.numericNABlanketFillValue
      )
      .setNAFillMode(_mainConfig.fillConfig.naFillMode)

    val (naFilledDataFrame, fillMap, detectedModelType) =
      if (_mainConfig.naFillFlag) {
        naConfig.generateCleanData()
      } else {
        (
          data,
          NaFillConfig(Map("" -> ""), Map("" -> 0.0), Map("" -> false)),
          naConfig.decideModel()
        )
      }

    val naLog: String = if (_mainConfig.naFillFlag) {
      s"NA values filled on Dataframe. Detected Model Type: $detectedModelType"
    } else {
      s"Detected Model Type: $detectedModelType"
    }

    logger.log(Level.INFO, naLog)
    println(naLog)

    (naFilledDataFrame, fillMap, detectedModelType)

  }

  private def varianceFilter(data: DataFrame): DataPrepReturn = {

    // Output has no feature vector
    val varianceFiltering = new VarianceFiltering(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .setDateTimeConversionType(_mainConfig.dateTimeConversionType)
      .setParallelism(_mainConfig.dataPrepParallelism)

    val (varianceFilteredData, removedColumns) =
      varianceFiltering.filterZeroVariance(_mainConfig.fieldsToIgnoreInVector)

    val varianceFilterLog = if (removedColumns.length == 0) {
      "Zero Variance fields have been removed from the data."
    } else {
      s"The following columns were removed due to zero variance: ${removedColumns.mkString(", ")}"
    }

    logger.log(Level.INFO, varianceFilterLog)
    println(varianceFilterLog)

    DataPrepReturn(varianceFilteredData, removedColumns)

  }

  private def outlierFilter(data: DataFrame): OutlierFilteringReturn = {

    // Output has no feature vector
    val outlierFiltering = new OutlierFiltering(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFilterBounds(_mainConfig.outlierConfig.filterBounds)
      .setLowerFilterNTile(_mainConfig.outlierConfig.lowerFilterNTile)
      .setUpperFilterNTile(_mainConfig.outlierConfig.upperFilterNTile)
      .setFilterPrecision(_mainConfig.outlierConfig.filterPrecision)
      .setParallelism(_mainConfig.dataPrepParallelism)
      .setContinuousDataThreshold(
        _mainConfig.outlierConfig.continuousDataThreshold
      )

    val (outlierCleanedData, outlierRemovedData, filteringMap) =
      outlierFiltering.filterContinuousOutliers(
        _mainConfig.fieldsToIgnoreInVector,
        _mainConfig.outlierConfig.fieldsToIgnore
      )

    val outlierRemovalInfo =
      s"Removed outlier data.  Total rows removed = ${outlierRemovedData.count()}"

    logger.log(Level.INFO, outlierRemovalInfo)
    println(outlierRemovalInfo)

    OutlierFilteringReturn(outlierCleanedData, filteringMap)

  }

  private def covarianceFilter(data: DataFrame,
                               fields: Array[String]): DataPrepReturn = {

    // Output has no feature vector

    val covarianceFilteredData = new FeatureCorrelationDetection(data, fields)
      .setLabelCol(_mainConfig.labelCol)
      .setParallelism(_mainConfig.dataPrepParallelism)
      .setCorrelationCutoffLow(
        _mainConfig.covarianceConfig.correlationCutoffLow
      )
      .setCorrelationCutoffHigh(
        _mainConfig.covarianceConfig.correlationCutoffHigh
      )
      .filterFeatureCorrelation()

    val removedFields =
      fieldRemovalCompare(fields, covarianceFilteredData.schema.fieldNames)

    val covarianceFilterLog =
      s"Covariance Filtering completed.\n  Removed fields: ${removedFields.mkString(", ")}"

    logger.log(Level.INFO, covarianceFilterLog)
    println(covarianceFilterLog)

    DataPrepReturn(covarianceFilteredData, removedFields.toArray)

  }

  private def pearsonFilter(data: DataFrame,
                            fields: Array[String],
                            modelType: String): DataPrepReturn = {

    // Requires a Dataframe that has a feature vector field.  Output has no feature vector.

    val pearsonFiltering = new PearsonFiltering(data, fields, modelType)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setFilterStatistic(_mainConfig.pearsonConfig.filterStatistic)
      .setFilterDirection(_mainConfig.pearsonConfig.filterDirection)
      .setFilterManualValue(_mainConfig.pearsonConfig.filterManualValue)
      .setFilterMode(_mainConfig.pearsonConfig.filterMode)
      .setAutoFilterNTile(_mainConfig.pearsonConfig.autoFilterNTile)
      .setParallelism(_mainConfig.dataPrepParallelism)
      .filterFields(_mainConfig.fieldsToIgnoreInVector)

    val removedFields =
      fieldRemovalCompare(fields, pearsonFiltering.schema.fieldNames)

    val pearsonFilterLog =
      s"Pearson Filtering completed.\n Removed fields: ${removedFields.mkString(", ")}"

    logger.log(Level.INFO, pearsonFiltering)
    println(pearsonFilterLog)

    DataPrepReturn(pearsonFiltering, removedFields.toArray)

  }

  private def scaler(data: DataFrame): DataFrame = {

    // Requires a Dataframe that has a feature vector field.  Output has a feature vector.

    val scaledData = new Scaler(data)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setScalerType(_mainConfig.scalingConfig.scalerType)
      .setScalerMin(_mainConfig.scalingConfig.scalerMin)
      .setScalerMax(_mainConfig.scalingConfig.scalerMax)
      .setStandardScalerMeanMode(
        _mainConfig.scalingConfig.standardScalerMeanFlag
      )
      .setStandardScalerStdDevMode(
        _mainConfig.scalingConfig.standardScalerStdDevFlag
      )
      .setPNorm(_mainConfig.scalingConfig.pNorm)
      .scaleFeatures()

    val scaleLog =
      s"Scaling of type '${_mainConfig.scalingConfig.scalerType}' completed."

    logger.log(Level.INFO, scaleLog)
    println(scaleLog)

    scaledData

  }

  def prepData(): DataGeneration = {

    // Record the Switch Settings from MainConfig to return an InferenceSwitchSettings object
    val inferenceSwitchSettings = recordInferenceSwitchSettings(_mainConfig)
    InferenceConfig.setInferenceSwitchSettings(inferenceSwitchSettings)

    // Perform validation of mlflow logging location so that it can fail early in case logging doesn't work.
    // Only run this if mlflow Logging flag is turned on.
    if (_mainConfig.mlFlowLoggingFlag) {
      val dirValidate = WorkspaceDirectoryValidation(
        _mainConfig.mlFlowConfig.mlFlowTrackingURI,
        _mainConfig.mlFlowConfig.mlFlowAPIToken,
        _mainConfig.mlFlowConfig.mlFlowExperimentName
      )
      if (dirValidate) {
        val rgx = "(\\/\\w+$)".r
        val dir =
          rgx.replaceFirstIn(_mainConfig.mlFlowConfig.mlFlowExperimentName, "")
        println(
          s"MLFlow Logging Directory confirmed accessible at: " +
            s"$dir"
        )
      }
    }

    val includeFieldsFinalData = _mainConfig.fieldsToIgnoreInVector

    println(
      s"Fields Set To Ignore: ${_mainConfig.fieldsToIgnoreInVector.mkString(", ")}"
    )

    val cacheLevel = StorageLevel.MEMORY_AND_DISK
    val unpersistBlock = true

    // log the settings used for the run
    logConfig()

    if (_mainConfig.dataPrepCachingFlag) {
      // cache the main DataFrame
      df.persist(cacheLevel)
      // force the cache
      df.count()
    }

    //DEBUG
    logger.log(Level.DEBUG, printSchema(df, "input").toString)

    val (naFilledData, fillMap, detectedModelType) = fillNA(df)
    val (entryPointData, entryPointFields, selectFields) =
      vectorPipeline(naFilledData, _mainConfig.fillConfig.cardinalitySwitch)

    // Record the Inference Settings for DataConfig
    val inferenceDataConfig =
      recordInferenceDataConfig(_mainConfig, selectFields)
    InferenceConfig.setInferenceDataConfig(inferenceDataConfig)

    val dataStage1 = entryPointData.select(selectFields map col: _*)

    // Record the Inference Settings for NaFillConfig mappings
    InferenceConfig.setInferenceNaFillConfig(
      fillMap.categoricalColumns,
      fillMap.numericColumns,
      fillMap.booleanColumns
    )

    // uncache the main DataFrame, force the GC
    val (persistDataStage1, dataStage1RowCount) =
      if (_mainConfig.dataPrepCachingFlag && _mainConfig.naFillFlag) {
        dataPersist(df, dataStage1, cacheLevel, unpersistBlock)
      } else {
        (dataStage1, "no count when data prep caching is disabled")
      }

    if (_mainConfig.naFillFlag) {
      println(dataStage1RowCount)
      logger.log(Level.INFO, dataStage1RowCount)
    }

    //DEBUG
    logger.log(Level.DEBUG, printSchema(dataStage1, "stage1").toString)
    logger.log(Level.DEBUG, printSchema(selectFields, "stage1_full").toString)

    // Variance Filtering
    val dataStage2 =
      if (_mainConfig.varianceFilterFlag) varianceFilter(persistDataStage1)
      else DataPrepReturn(persistDataStage1, Array.empty[String])

    // Record the Inference Settings for Variance Filtering
    InferenceConfig.setInferenceVarianceFilterConfig(dataStage2.fieldListing)

    val (persistDataStage2, dataStage2RowCount) =
      if (_mainConfig.dataPrepCachingFlag && _mainConfig.varianceFilterFlag) {
        dataPersist(
          persistDataStage1,
          dataStage2.outputData,
          cacheLevel,
          unpersistBlock
        )
      } else {
        (dataStage2.outputData, "no count when data prep caching is disabled")
      }

    if (_mainConfig.varianceFilterFlag) {
      println(dataStage2RowCount)
      logger.log(Level.INFO, dataStage2RowCount)
    }

    //DEBUG
    logger.log(
      Level.DEBUG,
      printSchema(dataStage2.outputData, "stage2").toString
    )

    // Outlier Filtering
    val dataStage3 =
      if (_mainConfig.outlierFilterFlag) outlierFilter(persistDataStage2)
      else
        OutlierFilteringReturn(
          persistDataStage2,
          Map.empty[String, (Double, String)]
        )

    val (persistDataStage3, dataStage3RowCount) =
      if (_mainConfig.dataPrepCachingFlag && _mainConfig.outlierFilterFlag) {
        dataPersist(
          persistDataStage2,
          dataStage3.outputData,
          cacheLevel,
          unpersistBlock
        )
      } else {
        (dataStage3.outputData, "no count when data prep caching is disabled")
      }

    if (_mainConfig.outlierFilterFlag) {
      println(dataStage2RowCount)
      logger.log(Level.INFO, dataStage3RowCount)
    }

    //DEBUG
    logger.log(
      Level.DEBUG,
      printSchema(dataStage3.outputData, "stage3").toString
    )

    // Record the Inference Settings for Outlier Filtering
    InferenceConfig.setInferenceOutlierFilteringConfig(
      dataStage3.fieldRemovalMap
    )

    // Next stages require a feature vector
    val (featurizedData, initialFields, initialFullFields) =
      vectorPipeline(persistDataStage3, cardinalityFlag = false)

    // Ensure that the only fields in the DataFrame are the Individual Feature Columns, Label, and Exclusion Fields
    val featureFieldCleanup = initialFields ++ Array(_mainConfig.labelCol)

    val featurizedDataCleaned = if (_mainConfig.dataPrepCachingFlag) {
      dataPersist(
        persistDataStage3,
        featurizedData.select(featureFieldCleanup map col: _*),
        cacheLevel,
        unpersistBlock
      )._1
    } else {
      featurizedData.select(featureFieldCleanup map col: _*)
    }

    //DEBUG
    logger.log(
      Level.DEBUG,
      printSchema(featurizedDataCleaned, "featurizedDataCleaned").toString
    )

    // Covariance Filtering
    val dataStage4 = if (_mainConfig.covarianceFilteringFlag) {
      covarianceFilter(featurizedDataCleaned, initialFields)
    } else DataPrepReturn(featurizedDataCleaned, Array.empty[String])

    val (persistDataStage4, dataStage4RowCount) =
      if (_mainConfig.dataPrepCachingFlag && _mainConfig.covarianceFilteringFlag) {
        dataPersist(
          featurizedDataCleaned,
          dataStage4.outputData,
          cacheLevel,
          unpersistBlock
        )
      } else {
        (dataStage4.outputData, "no count when data prep caching is disabled")
      }

    if (_mainConfig.covarianceFilteringFlag) {
      println(dataStage4RowCount)
      logger.log(Level.INFO, dataStage4RowCount)
    }

    //DEBUG
    logger.log(
      Level.DEBUG,
      printSchema(dataStage4.outputData, "stage4").toString
    )

    // Record the Inference Settings for Covariance Filtering
    InferenceConfig.setInferenceCovarianceFilteringConfig(
      dataStage4.fieldListing
    )

    // All stages after this point require a feature vector.
    val (dataStage5, stage5Fields, stage5FullFields) =
      vectorPipeline(persistDataStage4, cardinalityFlag = false)

    val (persistDataStage5, dataStage5RowCount) =
      if (_mainConfig.dataPrepCachingFlag) {
        dataPersist(persistDataStage4, dataStage5, cacheLevel, unpersistBlock)
      } else {
        (dataStage5, "no count when data prep caching is disabled")
      }

    // Pearson Filtering (generates a vector features Field)
    val (dataStage6, stage6Fields, stage6FullFields) =
      if (_mainConfig.pearsonFilteringFlag) {

        val pearsonReturn =
          pearsonFilter(persistDataStage5, stage5Fields, detectedModelType)

        // Record the Inference Settings for Pearson Filtering
        InferenceConfig.setInferencePearsonFilteringConfig(
          pearsonReturn.fieldListing
        )

        vectorPipeline(pearsonReturn.outputData, cardinalityFlag = false)
      } else {
        // Record the Inference Settings for Pearson Filtering
        InferenceConfig.setInferencePearsonFilteringConfig(Array.empty[String])
        (persistDataStage5, stage5Fields, stage5FullFields)
      }

    val (persistDataStage6, dataStage6RowCount) =
      if (_mainConfig.dataPrepCachingFlag && _mainConfig.pearsonFilteringFlag) {
        dataPersist(persistDataStage5, dataStage6, cacheLevel, unpersistBlock)
      } else {
        (dataStage5, "no count when data prep caching is disabled")
      }

    //DEBUG
    logger.log(Level.DEBUG, printSchema(persistDataStage6, "stage6").toString)

    // Feature Interaction Stage
    val featureInteractionResult = if (_mainConfig.featureInteractionFlag) {

      val nominalFields = stage6Fields
        .filter(x => x.takeRight(3) == "_si")
        .filterNot(x => x.contains(_labelCol))

      val continuousFields = stage6Fields
        .diff(nominalFields)
        .filterNot(_.contains(_labelCol))
        .filterNot(_.contains(_featuresCol))

      FeatureInteraction.interactFeatures(
        persistDataStage6,
        nominalFields,
        continuousFields,
        detectedModelType,
        _mainConfig.featureInteractionConfig.retentionMode,
        _labelCol,
        _featuresCol,
        _mainConfig.featureInteractionConfig.continuousDiscretizerBucketCount,
        _mainConfig.featureInteractionConfig.parallelism,
        _mainConfig.featureInteractionConfig.targetInteractionPercentage
      )
    } else {
      FeatureInteractionOutputPayload(
        persistDataStage6,
        stage6Fields,
        Array[InteractionPayloadExtract]()
      )
    }

    // Log the Inference config elements for Feature Interactions
    InferenceConfig.setFeatureInteractionConfig(
      FeatureInteractionConfig(featureInteractionResult.interactionReport)
    )

    val (persistFeatureInteractionData, persistFeatureInteractionCount) =
      if (_mainConfig.dataPrepCachingFlag && _mainConfig.featureInteractionFlag) {
        dataPersist(
          persistDataStage6,
          featureInteractionResult.data,
          cacheLevel,
          unpersistBlock
        )
      } else {
        (
          featureInteractionResult.data,
          "no count when data prep caching is disabled"
        )
      }

    //DEBUG
    logger.log(
      Level.DEBUG,
      printSchema(persistFeatureInteractionData, "featureInteractionStage").toString
    )

    // OneHotEncoding Option
    val (dataStage65, stage65Fields, stage65FullFields) =
      if (_mainConfig.oneHotEncodeFlag) {
        oneHotEncodeVector(
          persistFeatureInteractionData,
          featureInteractionResult.fullFeatureVectorColumns,
          persistFeatureInteractionData.schema.names
        )
      } else
        (
          persistFeatureInteractionData,
          featureInteractionResult.fullFeatureVectorColumns,
          persistFeatureInteractionData.schema.names
        )

    val (persistDataStage65, dataStage65RowCount) =
      if (_mainConfig.dataPrepCachingFlag && _mainConfig.oneHotEncodeFlag) {
        dataPersist(
          persistFeatureInteractionData,
          dataStage65,
          cacheLevel,
          unpersistBlock
        )
      } else {
        (dataStage65, "no count when data prep caching is disabled")
      }

    //DEBUG
    logger.log(Level.DEBUG, printSchema(persistDataStage65, "stage65").toString)

    // Scaler
    val dataStage7 =
      if (_mainConfig.scalingFlag) scaler(dataStage65) else dataStage65

    val (persistDataStage7, dataStage7RowCount) =
      if (_mainConfig.dataPrepCachingFlag && _mainConfig.scalingFlag) {
        dataPersist(persistDataStage65, dataStage7, cacheLevel, unpersistBlock)
      } else {
        (dataStage7, "no count when data prep caching is disabled")
      }

    if (_mainConfig.scalingFlag && _mainConfig.dataPrepCachingFlag) {
      println(dataStage7RowCount)
      logger.log(Level.INFO, dataStage7RowCount)
    }

    // Record the Inference Settings for Scaling
    InferenceConfig.setInferenceScalingConfig(_mainConfig.scalingConfig)

    // Get the final DataFrame Field Loading

    val finalStageDF =
      persistDataStage7.select(persistDataStage7.columns map col: _*)
    if (_mainConfig.dataPrepCachingFlag)
      dataPersist(persistDataStage7, finalStageDF, cacheLevel, unpersistBlock)
    else finalStageDF.persist(cacheLevel)

    val finalCount = finalStageDF.count

    val finalSchema = s"Final Schema: \n    ${stage65Fields.mkString(", ")}"
    val finalFullSchema =
      s"Final Full Schema: \n    ${finalStageDF.columns.mkString(", ")}"

    val finalOutputDataFrame1 =
      if (_mainConfig.geneticConfig.trainSplitMethod == "kSample") {
        SyntheticFeatureGenerator(
          finalStageDF,
          _mainConfig.featuresCol,
          _mainConfig.labelCol,
          _mainConfig.geneticConfig.kSampleConfig.syntheticCol,
          _mainConfig.fieldsToIgnoreInVector,
          _mainConfig.geneticConfig.kSampleConfig.kGroups,
          _mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter,
          _mainConfig.geneticConfig.kSampleConfig.kMeansTolerance,
          _mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement,
          _mainConfig.geneticConfig.kSampleConfig.kMeansSeed,
          _mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol,
          _mainConfig.geneticConfig.kSampleConfig.lshHashTables,
          _mainConfig.geneticConfig.kSampleConfig.lshSeed,
          _mainConfig.geneticConfig.kSampleConfig.lshOutputCol,
          _mainConfig.geneticConfig.kSampleConfig.quorumCount,
          _mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate,
          _mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod,
          _mainConfig.geneticConfig.kSampleConfig.mutationMode,
          _mainConfig.geneticConfig.kSampleConfig.mutationValue,
          _mainConfig.geneticConfig.kSampleConfig.labelBalanceMode,
          _mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold,
          _mainConfig.geneticConfig.kSampleConfig.numericRatio,
          _mainConfig.geneticConfig.kSampleConfig.numericTarget
        )
      } else finalStageDF

    // If scaling is used, make sure that the synthetic data has the same scaling.
    val finalOutputDataFrame2 =
      if (_mainConfig.scalingFlag & _mainConfig.geneticConfig.trainSplitMethod == "kSample") {
        val syntheticData = finalOutputDataFrame1.filter(
          col(_mainConfig.geneticConfig.kSampleConfig.syntheticCol)
        )
        scaler(syntheticData).union(
          finalOutputDataFrame1.filter(
            col(_mainConfig.geneticConfig.kSampleConfig.syntheticCol) === false
          )
        )
      } else finalOutputDataFrame1

    val finalStatement =
      s"Data Prep complete.  Final Dataframe cached. Total Observations: $finalCount"
    // DEBUG
    logger.log(Level.INFO, finalSchema)
    logger.log(Level.INFO, finalFullSchema)
    logger.log(Level.INFO, finalStatement)
    println(finalStatement)

    DataGeneration(
      finalOutputDataFrame2,
      finalOutputDataFrame2.columns,
      detectedModelType
    )

  }

}
