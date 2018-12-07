package com.databricks.spark.automatedml.executor

import com.databricks.spark.automatedml.params.DataPrepConfig
import com.databricks.spark.automatedml.pipeline.FeaturePipeline
import com.databricks.spark.automatedml.sanitize._
import com.databricks.spark.automatedml.utils.AutomationTools
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

class DataPrep(df: DataFrame) extends AutomationConfig with AutomationTools {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _dataPrepFlags = _dataPrepConfigDefaults

  def setDataPrepFlags(value: DataPrepConfig): this.type = {
    _dataPrepFlags = value
    this
  }

  def getDataPrepFlags: DataPrepConfig = _dataPrepFlags

  private def logConfig(): Unit = {

    val configString = s"Configuration setting flags: \n NA Fill Flag: ${_dataPrepFlags.naFillFlag.toString}" +
      s"\n Zero Variance Filter Flag: ${_dataPrepFlags.varianceFilterFlag.toString}" +
      s"\n Outlier Filter Flag: ${_dataPrepFlags.outlierFilterFlag.toString}" +
      s"\n Covariance Filter Flag: ${_dataPrepFlags.covarianceFilterFlag.toString}" +
      s"\n Pearson Filter Flag: ${_dataPrepFlags.pearsonFilterFlag.toString}" +
      s"\n Scaling Flag: ${_dataPrepFlags.scalingFlag.toString}"

    println(configString)
    logger.log(Level.INFO, configString)

  }

  private def vectorPipeline(data: DataFrame): (DataFrame, Array[String]) = {

    // Creates the feature vector and returns the fields that go into the vector

    new FeaturePipeline(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .makeFeaturePipeline()

  }

  private def fillNA(data: DataFrame): (DataFrame, String) = {

    // Output has no feature vector

    val naConfig = new DataSanitizer(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .setModelSelectionDistinctThreshold(_mainConfig.fillConfig.modelSelectionDistinctThreshold)
      .setNumericFillStat(_mainConfig.fillConfig.numericFillStat)
      .setCharacterFillStat(_mainConfig.fillConfig.characterFillStat)

    val (naFilledDataFrame, detectedModelType) = if (_dataPrepFlags.naFillFlag) {
      naConfig.generateCleanData()
    } else {
      (data, naConfig.decideModel())
    }

    val naLog: String = if (_dataPrepFlags.naFillFlag) {
      s"NA values filled on Dataframe. Detected Model Type: $detectedModelType"
    } else {
      s"Detected Model Type: $detectedModelType"
    }

    logger.log(Level.INFO, naLog)
    println(naLog)

    (naFilledDataFrame, detectedModelType)

  }

  private def varianceFilter(data: DataFrame): DataFrame = {

    // Output has no feature vector
    val varianceFiltering = new VarianceFiltering(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)

    val varianceFilteredData = varianceFiltering.filterZeroVariance()

    val varianceFilterLog = "Zero Variance fields have been removed from the data."

    logger.log(Level.INFO, varianceFilterLog)
    println(varianceFilterLog)

    varianceFilteredData

  }

  private def outlierFilter(data: DataFrame): DataFrame = {

    // Output has no feature vector

    val outlierFiltering = new OutlierFiltering(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFilterBounds(_mainConfig.outlierConfig.filterBounds)
      .setLowerFilterNTile(_mainConfig.outlierConfig.lowerFilterNTile)
      .setUpperFilterNTile(_mainConfig.outlierConfig.upperFilterNTile)
      .setFilterPrecision(_mainConfig.outlierConfig.filterPrecision)
      .setContinuousDataThreshold(_mainConfig.outlierConfig.continuousDataThreshold)

    val (outlierCleanedData, outlierRemovedData) = outlierFiltering.filterContinuousOutliers(
      _mainConfig.outlierConfig.fieldsToIgnore
    )

    val outlierRemovalInfo = s"Removed outlier data.  Total rows removed = ${outlierRemovedData.count()}"

    logger.log(Level.INFO, outlierRemovalInfo)
    println(outlierRemovalInfo)

    outlierCleanedData

  }

  private def covarianceFilter(data: DataFrame, fields: Array[String]): DataFrame = {

    // Output has no feature vector

    val covarianceFilteredData = new FeatureCorrelationDetection(data, fields)
      .setLabelCol(_mainConfig.labelCol)
      .setCorrelationCutoffLow(_mainConfig.covarianceConfig.correlationCutoffLow)
      .setCorrelationCutoffHigh(_mainConfig.covarianceConfig.correlationCutoffHigh)
      .filterFeatureCorrelation()

    val removedFields = fieldRemovalCompare(fields, covarianceFilteredData.schema.fieldNames)

    val covarianceFilterLog = s"Covariance Filtering completed.\n  Removed fields: ${removedFields.mkString(", ")}"

    logger.log(Level.INFO, covarianceFilterLog)
    println(covarianceFilterLog)

    covarianceFilteredData

  }

  private def pearsonFilter(data: DataFrame, fields: Array[String]): DataFrame = {

    // Requires a Dataframe that has a feature vector field.  Output has no feature vector.

    val pearsonFiltering = new PearsonFiltering(data, fields)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setFilterStatistic(_mainConfig.pearsonConfig.filterStatistic)
      .setFilterDirection(_mainConfig.pearsonConfig.filterDirection)
      .setFilterManualValue(_mainConfig.pearsonConfig.filterManualValue)
      .setFilterMode(_mainConfig.pearsonConfig.filterMode)
      .setAutoFilterNTile(_mainConfig.pearsonConfig.autoFilterNTile)
      .filterFields()

    val removedFields = fieldRemovalCompare(fields, pearsonFiltering.schema.fieldNames)

    val pearsonFilterLog = s"Pearson Filtering completed.\n Removed fields: ${removedFields.mkString(", ")}"

    logger.log(Level.INFO, pearsonFiltering)
    println(pearsonFilterLog)

    pearsonFiltering

  }

  private def scaler(data: DataFrame): DataFrame = {

    // Requires a Dataframe that has a feature vector field.  Output has a feature vector.

    val scaledData = new Scaler(data)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setScalerType(_mainConfig.scalingConfig.scalerType)
      .setScalerMin(_mainConfig.scalingConfig.scalerMin)
      .setScalerMax(_mainConfig.scalingConfig.scalerMax)
      .setStandardScalerMeanMode(_mainConfig.scalingConfig.standardScalerMeanFlag)
      .setStandardScalerStdDevMode(_mainConfig.scalingConfig.standardScalerStdDevFlag)
      .setPNorm(_mainConfig.scalingConfig.pNorm)
      .scaleFeatures()

    val scaleLog = s"Scaling of type '${_mainConfig.scalingConfig.scalerType}' completed."

    logger.log(Level.INFO, scaleLog)
    println(scaleLog)

    scaledData

  }

  def prepData(): (DataFrame, Array[String], String) = {


    //TODO: make these persist/cache statement configurable based on the size of the training set.
    val cacheLevel = StorageLevel.MEMORY_AND_DISK
    val unpersistBlock = true

    // log the settings used for the run
    logConfig()

    // cache the main DataFrame
    df.persist(cacheLevel)

    val (dataStage1, detectedModelType) = fillNA(df)

    // uncache the main DataFrame, force the GC
    dataPersist(df, dataStage1, cacheLevel, unpersistBlock)

    // Variance Filtering
    val dataStage2 = if (_dataPrepFlags.varianceFilterFlag) varianceFilter(dataStage1) else dataStage1

    dataPersist(dataStage1, dataStage2, cacheLevel, unpersistBlock)

    // Outlier Filtering
    val dataStage3 = if (_dataPrepFlags.outlierFilterFlag) outlierFilter(dataStage2) else dataStage2

    dataPersist(dataStage2, dataStage3, cacheLevel, unpersistBlock)

    // Next stages require a feature vector
    val (featurizedData, initialFields) = vectorPipeline(dataStage3)

    // Ensure that the only fields in the DataFrame are the Individual Feature Columns and the Label Column
    val featureFieldCleanup = initialFields ++ Array(_mainConfig.labelCol)

    val featurizedDataCleaned = featurizedData.select(featureFieldCleanup map col: _*)

    dataPersist(dataStage3, featurizedDataCleaned, cacheLevel, unpersistBlock)

    // Covariance Filtering
    val dataStage4 = if (_dataPrepFlags.covarianceFilterFlag) {
      covarianceFilter(featurizedDataCleaned, initialFields)
    } else featurizedDataCleaned

    dataPersist(featurizedDataCleaned, dataStage4, cacheLevel, unpersistBlock)

    // All stages after this point require a feature vector.
    val (dataStage5, stage5Fields) = vectorPipeline(dataStage4)

    dataPersist(dataStage4, dataStage5, cacheLevel, unpersistBlock)

    // Pearson Filtering (generates a vector features Field)
    val (dataStage6, stage6Fields) = if (_dataPrepFlags.pearsonFilterFlag) {
      vectorPipeline(pearsonFilter(dataStage5, stage5Fields))
    } else (dataStage5, stage5Fields)

    // Scaler
    val dataStage7 = if (_dataPrepFlags.scalingFlag) scaler(dataStage6) else dataStage6

    dataPersist(dataStage5, dataStage7, cacheLevel, unpersistBlock)

    val finalStatement = s"Data Prep complete.  Final Dataframe cached."

    logger.log(Level.INFO, finalStatement)
    println(finalStatement)

    (dataStage7, stage6Fields, detectedModelType)

  }

}
