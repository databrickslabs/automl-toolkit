package com.databricks.spark.automatedml.executor

import com.databricks.spark.automatedml.pipeline.FeaturePipeline
import com.databricks.spark.automatedml.sanitize._
import com.databricks.spark.automatedml.utils.AutomationTools
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

class DataPrep(df: DataFrame) extends AutomationConfig with AutomationTools {

  //    TODO: parallelism config for non genetic parallel control should be added
  private val logger: Logger = Logger.getLogger(this.getClass)

  private def logConfig(): Unit = {

    val configString = s"Configuration setting flags: \n NA Fill Flag: ${_mainConfig.naFillFlag.toString}" +
      s"\n Zero Variance Filter Flag: ${_mainConfig.varianceFilterFlag.toString}" +
      s"\n Outlier Filter Flag: ${_mainConfig.outlierFilterFlag.toString}" +
      s"\n Covariance Filter Flag: ${_mainConfig.covarianceFilteringFlag.toString}" +
      s"\n Pearson Filter Flag: ${_mainConfig.pearsonFilteringFlag.toString}" +
      s"\n Scaling Flag: ${_mainConfig.scalingFlag.toString}" +
      s"\n MlFlow Logging Flag: ${_mainConfig.mlFlowLoggingFlag.toString}" +
      s"\n Early Stopping Flag: ${_mainConfig.autoStoppingFlag.toString}"

    println(configString)
    logger.log(Level.INFO, configString)

  }

  private def vectorPipeline(data: DataFrame): (DataFrame, Array[String], Array[String]) = {

    // Creates the feature vector and returns the fields that go into the vector

    new FeaturePipeline(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .setDateTimeConversionType(_mainConfig.dateTimeConversionType)
      .makeFeaturePipeline(_mainConfig.fieldsToIgnoreInVector)

  }

  private def fillNA(data: DataFrame): (DataFrame, String) = {

    // Output has no feature vector

    val naConfig = new DataSanitizer(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .setModelSelectionDistinctThreshold(_mainConfig.fillConfig.modelSelectionDistinctThreshold)
      .setNumericFillStat(_mainConfig.fillConfig.numericFillStat)
      .setCharacterFillStat(_mainConfig.fillConfig.characterFillStat)
      .setParallelism(_mainConfig.geneticConfig.parallelism)
      .setFieldsToIgnoreInVector(_mainConfig.fieldsToIgnoreInVector)

    val (naFilledDataFrame, detectedModelType) = if (_mainConfig.naFillFlag) {
      naConfig.generateCleanData()
    } else {
      (data, naConfig.decideModel())
    }

    val naLog: String = if (_mainConfig.naFillFlag) {
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
      .setDateTimeConversionType(_mainConfig.dateTimeConversionType)

    val (varianceFilteredData, removedColumns) = varianceFiltering.filterZeroVariance(_mainConfig.fieldsToIgnoreInVector)

    val varianceFilterLog = if (removedColumns.length == 0) {
      "Zero Variance fields have been removed from the data."
    } else {
      s"The following columns were removed due to zero variance: ${removedColumns.mkString(", ")}"
    }

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
      .setParallelism(_mainConfig.geneticConfig.parallelism)
      .setContinuousDataThreshold(_mainConfig.outlierConfig.continuousDataThreshold)

    val (outlierCleanedData, outlierRemovedData) = outlierFiltering.filterContinuousOutliers(
      _mainConfig.fieldsToIgnoreInVector, _mainConfig.outlierConfig.fieldsToIgnore
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
      .filterFields(_mainConfig.fieldsToIgnoreInVector)

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


    //TODO: add in a verbosity toggle to print / log certain elements
    // TODO: should have "full logging", "standard logging" and "log4j only"

    val includeFieldsFinalData = _mainConfig.fieldsToIgnoreInVector

    println(s"Fields Set To Ignore: ${_mainConfig.fieldsToIgnoreInVector.mkString(", ")}")

    val cacheLevel = StorageLevel.MEMORY_AND_DISK
    val unpersistBlock = true

    // log the settings used for the run
    logConfig()

    // cache the main DataFrame
    df.persist(cacheLevel)
    // force the cache
    df.count()

    //DEBUG
    logger.log(Level.DEBUG, printSchema(df, "input").toString)


    // Start by converting fields
    val (entryPointDf, entryPointFields, selectFields) = vectorPipeline(df)

    // up to here behaves correctly.

    logger.log(Level.DEBUG, printSchema(entryPointDf, "entryPoint").toString)

    //val restrictFields = entryPointFields ++ List(_mainConfig.labelCol)

    val entryPointDataRestrict = entryPointDf.select(selectFields map col:_*)


    // this ignores the fieldsToIgnore and reparses the date and time fields.  FIXED.
    val (dataStage1, detectedModelType) = fillNA(entryPointDataRestrict)

    // uncache the main DataFrame, force the GC
    val (persistDataStage1, dataStage1RowCount) = dataPersist(df, dataStage1, cacheLevel, unpersistBlock)

    // TODO: add logging flag switch for this
    println(dataStage1RowCount)
    logger.log(Level.INFO, dataStage1RowCount)

    //DEBUG
    logger.log(Level.DEBUG, printSchema(dataStage1, "stage1").toString)
    logger.log(Level.DEBUG, printSchema(selectFields, "stage1_full").toString)

    // Variance Filtering
    val dataStage2 = if (_mainConfig.varianceFilterFlag) varianceFilter(persistDataStage1) else persistDataStage1

    val (persistDataStage2, dataStage2RowCount) = dataPersist(persistDataStage1, dataStage2, cacheLevel, unpersistBlock)

    println(dataStage2RowCount)
    logger.log(Level.INFO, dataStage2RowCount)

    //DEBUG

    logger.log(Level.DEBUG, printSchema(dataStage2, "stage2").toString)

    // Outlier Filtering
    val dataStage3 = if (_mainConfig.outlierFilterFlag) outlierFilter(persistDataStage2) else persistDataStage2

    val (persistDataStage3, dataStage3RowCount) = dataPersist(persistDataStage2, dataStage3, cacheLevel, unpersistBlock)

    println(dataStage2RowCount)
    logger.log(Level.INFO, dataStage3RowCount)

    //DEBUG

    logger.log(Level.DEBUG, printSchema(dataStage3, "stage3").toString)

    // Next stages require a feature vector
    val (featurizedData, initialFields, initialFullFields) = vectorPipeline(persistDataStage3)

    // Ensure that the only fields in the DataFrame are the Individual Feature Columns, Label, and Exclusion Fields
    val featureFieldCleanup = initialFields ++ Array(_mainConfig.labelCol) ++ includeFieldsFinalData

    val featurizedDataCleaned = featurizedData.select(featureFieldCleanup map col: _*)

    val (persistFeaturizedDataCleaned, featurizedDataCleanedRowCount) = dataPersist(persistDataStage3, featurizedDataCleaned, cacheLevel, unpersistBlock)

    println(featurizedDataCleanedRowCount)
    logger.log(Level.INFO, featurizedDataCleanedRowCount)

    //DEBUG

    logger.log(Level.DEBUG, printSchema(featurizedDataCleaned, "featurizedDataCleaned").toString)

    // Covariance Filtering
    val dataStage4 = if (_mainConfig.covarianceFilteringFlag) {
      covarianceFilter(persistFeaturizedDataCleaned, initialFields)
    } else persistFeaturizedDataCleaned

    val (persistDataStage4, dataStage4RowCount) = dataPersist(persistFeaturizedDataCleaned, dataStage4, cacheLevel, unpersistBlock)

    println(dataStage4RowCount)
    logger.log(Level.INFO, dataStage4RowCount)

    //DEBUG

    logger.log(Level.DEBUG, printSchema(dataStage4, "stage4").toString)

    // All stages after this point require a feature vector.
    val (dataStage5, stage5Fields, stage5FullFields) = vectorPipeline(persistDataStage4)

    val (persistDataStage5, dataStage5RowCount) = dataPersist(persistDataStage4, dataStage5, cacheLevel, unpersistBlock)

    println(dataStage5RowCount)
    logger.log(Level.INFO, dataStage5RowCount)

    //DEBUG

    logger.log(Level.DEBUG, printSchema(dataStage5, "stage5").toString)

    // Pearson Filtering (generates a vector features Field)
    val (dataStage6, stage6Fields, stage6FullFields) = if (_mainConfig.pearsonFilteringFlag) {
      vectorPipeline(pearsonFilter(persistDataStage5, stage5Fields))
    } else (persistDataStage5, stage5Fields, stage5FullFields)

    //DEBUG
    logger.log(Level.DEBUG, printSchema(dataStage6, "stage6").toString)

    // Scaler
    val dataStage7 = if (_mainConfig.scalingFlag) scaler(dataStage6) else dataStage6

    val (persistDataStage7, dataStage7RowCount) = dataPersist(persistDataStage5, dataStage7, cacheLevel, unpersistBlock)

    println(dataStage7RowCount)
    logger.log(Level.INFO, dataStage7RowCount)

    // This is making a mess in the output. Commenting out for now.
    // TODO: Determine if schema getter is necessary outside of logging
    val finalSchema = s"Final Schema: \n    ${stage6Fields.mkString(", ")}"
    val finalFullSchema = s"Final Full Schema: \n    ${stage6FullFields.mkString(", ")}"

    logger.log(Level.INFO, finalSchema)
//    println(finalSchema)

    logger.log(Level.INFO, finalFullSchema)
//    println(finalFullSchema)

    val finalStatement = s"Data Prep complete.  Final Dataframe cached."

    logger.log(Level.INFO, finalStatement)
    println(finalStatement)

    (persistDataStage7, stage6Fields, detectedModelType)

  }

}
