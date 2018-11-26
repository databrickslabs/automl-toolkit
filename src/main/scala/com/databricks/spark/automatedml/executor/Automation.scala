package com.databricks.spark.automatedml.executor

import com.databricks.spark.automatedml.pipeline.FeaturePipeline
import com.databricks.spark.automatedml.sanitize._
import com.databricks.spark.automatedml.utils.{AutomationTools, SparkSessionWrapper}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._



class Automation() extends AutomationConfig with SparkSessionWrapper with AutomationTools {

  require(_supportedModels.contains(_mainConfig.modelType))

  def dataPrep(data: DataFrame):(DataFrame, Array[String], String) = {

    // Fill na values
    val dataSanitizer = new DataSanitizer(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .setCharacterFillStat(_mainConfig.fillConfig.characterFillStat)
      .setNumericFillStat(_mainConfig.fillConfig.numericFillStat)
      .setModelSelectionDistinctThreshold(_mainConfig.fillConfig.modelSelectionDistinctThreshold)

    val (filledData, modelType) = if (_mainConfig.naFillFlag) {
      val (naFillData, detectedModelType) = dataSanitizer.generateCleanData()
      //TODO: add logging to log4j
      println(s"NA values filled on Dataframe. Detected Model Type: $detectedModelType")

      (naFillData, detectedModelType)
    } else {
      (data, dataSanitizer.decideModel())
    }

    // Variance Filtering
    val varianceFiltering = new VarianceFiltering(filledData)
      .setLabelCol(_mainConfig.labelCol)

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
      val (cleanedData, outlierData) = outlierFiltering.filterContinuousOutliers(_mainConfig.outlierConfig.fieldsToIgnore)
      //TODO: add logging to log4j
      println(s"Removed outlier data.  Total rows removed = ${outlierData.count()}")

      (cleanedData, outlierData)
    } else {
      (filledData, spark.createDataFrame(sc.emptyRDD[Row], filledData.schema))
    }

    val (preCovariance, preCovarianceFields) = new FeaturePipeline(outlierCleanedData).makeFeaturePipeline()

    val preCovarianceFieldsRestrict = preCovarianceFields ++ Array(_labelCol)
    val preCovarianceFilteredData = preCovariance.select(preCovarianceFieldsRestrict map col: _*)

    // Covariance Filtering
    val covarianceFiltering = new FeatureCorrelationDetection(preCovarianceFilteredData, preCovarianceFields)
      .setLabelCol(_mainConfig.labelCol)
      .setCorrelationCutoffLow(_mainConfig.covarianceConfig.correlationCutoffLow)
      .setCorrelationCutoffHigh(_mainConfig.covarianceConfig.correlationCutoffHigh)

    val (postFilteredData, postFilteredFields) = if (_mainConfig.covarianceFilteringFlag) {
      val covarianceFilter = covarianceFiltering.filterFeatureCorrelation()

      println(s"Post Covariance Filtered fields: '${covarianceFilter.schema.fieldNames.mkString(", ")}'")
      println(s"Row count: ${covarianceFilter.count()}")

      new FeaturePipeline(covarianceFiltering.filterFeatureCorrelation()).makeFeaturePipeline()
    } else {
      new FeaturePipeline(preCovarianceFilteredData).makeFeaturePipeline()
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
      new FeaturePipeline(pearsonFiltering.filterFields()).makeFeaturePipeline()
    } else {
      (postFilteredData, postFilteredFields)
    }
    println(s"Finished with Data Generation. Schema: ${outputData.schema}, fieldListing: '${
      fieldListing.mkString(", ")}'")
    (outputData, fieldListing, modelType)

  }

}
