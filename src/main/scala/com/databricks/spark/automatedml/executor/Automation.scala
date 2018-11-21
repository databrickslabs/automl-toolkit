package com.databricks.spark.automatedml.executor

import com.databricks.spark.automatedml.pipeline.FeaturePipeline
import com.databricks.spark.automatedml.sanitize._
import com.databricks.spark.automatedml.utils.{AutomationTools, SparkSessionWrapper}
import org.apache.spark.sql.{DataFrame, Row}



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
      dataSanitizer.generateCleanData()
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
      outlierFiltering.filterContinuousOutliers(_mainConfig.outlierConfig.fieldsToIgnore)
    } else {
      (filledData, spark.createDataFrame(sc.emptyRDD[Row], filledData.schema))
    }

    // Construct the Featurized Data Pipeline
    val (preFilteredData, preFilteredFields) = new FeaturePipeline(outlierCleanedData).makeFeaturePipeline()

    // Covariance Filtering //TODO: this is broken due to the "features" column being present.  FIX THIS. try the drop
    //TODO: this might need to get the original field list, the fields that were removed previously, and select only those.
    //TODO: this requires a bit of a redesign of how this chain of elements work.  Maintain a Buffer of fields to keep?
    val covarianceFiltering = new FeatureCorrelationDetection(preFilteredData.drop(_mainConfig.featuresCol), preFilteredFields)
      .setLabelCol(_mainConfig.labelCol)
      .setCorrelationCutoffLow(_mainConfig.covarianceConfig.correlationCutoffLow)
      .setCorrelationCutoffHigh(_mainConfig.covarianceConfig.correlationCutoffHigh)

    val (postFilteredData, postFilteredFields) = if (_mainConfig.covarianceFilteringFlag) {
      new FeaturePipeline(covarianceFiltering.filterFeatureCorrelation()).makeFeaturePipeline()
    } else {
      (preFilteredData, preFilteredFields)
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

    (outputData, fieldListing, modelType)

  }

}
