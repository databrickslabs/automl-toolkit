package com.databricks.spark.automatedml.executor

import com.databricks.spark.automatedml.pipeline.FeaturePipeline
import com.databricks.spark.automatedml.sanitize._
import com.databricks.spark.automatedml.utils.AutomationTools
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._

import org.apache.log4j.{Level, Logger}

class Automation extends AutomationConfig with AutomationTools {

  require(_supportedModels.contains(_mainConfig.modelFamily))

  val logger: Logger = Logger.getLogger(this.getClass)

  def dataPrep(data: DataFrame):(DataFrame, Array[String], String) = {

    // Print to stdout the filter settings
    println(s"Configuration setting flags: \n NA Fill Flag: ${_naFillFlag.toString} \n " +
      s"Zero Variance Filter Flag: ${_varianceFilterFlag.toString} \n  Outlier Filter Flag: " +
      s"${_outlierFilterFlag.toString}\n Covariance Filter Flag: " +
      s"${_covarianceFilterFlag.toString} \n Pearson Filter Flag: ${_pearsonFilterFlag.toString}")

    // Log the data Prep settings
    logger.log(Level.INFO, s"NA Fill Flag: ${_naFillFlag.toString}")
    logger.log(Level.INFO, s"Zero Variance Filter Flag: ${_varianceFilterFlag.toString}")
    logger.log(Level.INFO, s"Outlier Filter Flag: ${_outlierFilterFlag.toString}")
    logger.log(Level.INFO, s"Covariance Filter Flag: ${_covarianceFilterFlag.toString}")
    logger.log(Level.INFO, s"Pearson Filter Flag: ${_pearsonFilterFlag.toString}")

    // Fill na values
    val dataSanitizer = new DataSanitizer(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .setCharacterFillStat(_mainConfig.fillConfig.characterFillStat)
      .setNumericFillStat(_mainConfig.fillConfig.numericFillStat)
      .setModelSelectionDistinctThreshold(_mainConfig.fillConfig.modelSelectionDistinctThreshold)

    val (filledData, modelType) = if (_mainConfig.naFillFlag) {

      val (naFillData, detectedModelType) = dataSanitizer.generateCleanData()

      val naLog: String = s"NA values filled on Dataframe. Detected Model Type: $detectedModelType"
      logger.log(Level.INFO, naLog)
      println(naLog)

      (naFillData, detectedModelType)
    } else {
      (data, dataSanitizer.decideModel())
    }

    // Variance Filtering
    val varianceFiltering = new VarianceFiltering(filledData)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)

    val varianceFilteredData = if (_mainConfig.varianceFilterFlag) varianceFiltering.filterZeroVariance() else filledData


    // Outlier Filtering
    val outlierFiltering = new OutlierFiltering(varianceFilteredData)
      .setLabelCol(_mainConfig.labelCol)
      .setFilterBounds(_mainConfig.outlierConfig.filterBounds)
      .setLowerFilterNTile(_mainConfig.outlierConfig.lowerFilterNTile)
      .setUpperFilterNTile(_mainConfig.outlierConfig.upperFilterNTile)
      .setFilterPrecision(_mainConfig.outlierConfig.filterPrecision)
      .setContinuousDataThreshold(_mainConfig.outlierConfig.continuousDataThreshold)

    val (outlierCleanedData, removedData) = if (_mainConfig.outlierFilterFlag) {
      val (cleanedData, outlierData) = outlierFiltering.filterContinuousOutliers(
        _mainConfig.outlierConfig.fieldsToIgnore)

      val outlierRemovalInfo = s"Removed outlier data.  Total rows removed = ${outlierData.count()}"
      logger.log(Level.INFO, outlierRemovalInfo)
      println(outlierRemovalInfo)

      (cleanedData, outlierData)
    } else {
      (filledData, spark.createDataFrame(sc.emptyRDD[Row], filledData.schema))
    }

    val (preCovariance, preCovarianceFields) = new FeaturePipeline(outlierCleanedData)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .makeFeaturePipeline()

    val preCovarianceFieldsRestrict = preCovarianceFields ++ Array(_labelCol)
    val preCovarianceFilteredData = preCovariance.select(preCovarianceFieldsRestrict map col: _*)

    // Covariance Filtering
    val covarianceFiltering = new FeatureCorrelationDetection(preCovarianceFilteredData, preCovarianceFields)
      .setLabelCol(_mainConfig.labelCol)
      .setCorrelationCutoffLow(_mainConfig.covarianceConfig.correlationCutoffLow)
      .setCorrelationCutoffHigh(_mainConfig.covarianceConfig.correlationCutoffHigh)

    val (postFilteredData, postFilteredFields) = if (_mainConfig.covarianceFilteringFlag) {
      val covarianceFilter = covarianceFiltering.filterFeatureCorrelation()

      val covarianceFilteringInfo = s"Post Covariance Filtered fields: '${
        covarianceFilter.schema.fieldNames.mkString(", ")}'"
      logger.log(Level.INFO, covarianceFilteringInfo)
      println(covarianceFilteringInfo)

      new FeaturePipeline(covarianceFiltering.filterFeatureCorrelation())
        .setLabelCol(_mainConfig.labelCol)
        .setFeatureCol(_mainConfig.featuresCol)
        .makeFeaturePipeline()
    } else {
      new FeaturePipeline(preCovarianceFilteredData)
        .setLabelCol(_mainConfig.labelCol)
        .setFeatureCol(_mainConfig.featuresCol)
        .makeFeaturePipeline()
    }

    // Pearson Filtering
    val pearsonFiltering = new PearsonFiltering(postFilteredData, postFilteredFields)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setFilterStatistic(_mainConfig.pearsonConfig.filterStatistic)
      .setFilterDirection(_mainConfig.pearsonConfig.filterDirection)
      .setFilterManualValue(_mainConfig.pearsonConfig.filterManualValue)
      .setFilterMode(_mainConfig.pearsonConfig.filterMode)
      .setAutoFilterNTile(_mainConfig.pearsonConfig.autoFilterNTile)

    // Final Featurization and Field Listing Generation (method's output)
    val (outputData, fieldListing) = if (_mainConfig.pearsonFilteringFlag) {
      new FeaturePipeline(pearsonFiltering.filterFields())
        .setLabelCol(_mainConfig.labelCol)
        .setFeatureCol(_mainConfig.featuresCol)
        .makeFeaturePipeline()
    } else {
      (postFilteredData, postFilteredFields)
    }
    val finalFilteringInfo = s"Finished with Data Generation. \n  Schema: ${
      outputData.schema} \n  Feature Field Listing: '${fieldListing.mkString(", ")}'"
    println(finalFilteringInfo)
    logger.log(Level.INFO, finalFilteringInfo)

    (outputData, fieldListing, modelType)

  }

}
