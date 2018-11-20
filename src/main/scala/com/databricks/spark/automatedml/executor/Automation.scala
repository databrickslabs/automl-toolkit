package com.databricks.spark.automatedml.executor

import com.databricks.spark.automatedml.model.{MLPCTuner, RandomForestTuner}
import com.databricks.spark.automatedml.params._
import com.databricks.spark.automatedml.pipeline.FeaturePipeline
import com.databricks.spark.automatedml.sanitize._
import com.databricks.spark.automatedml.utils.SparkSessionWrapper
import org.apache.spark.sql.{DataFrame, Row}


case class ModelingConfig(
                                   labelCol: String,
                                   featuresCol: String,
                                   numericBoundaries: Map[String, (Double, Double)],
                                   stringBoundaries: Map[String, List[String]],
                                   scoringMetric: String
                                 )

sealed class Automation(config: MainConfig) extends Defaults with SparkSessionWrapper {

  require(_supportedModels.contains(config.modelType))

  val _geneticConfig: GeneticConfig = config.geneticConfig.getOrElse(_geneticTunerDefaults)
  val _scoringOptimizationStrategy: String = config.scoringOptimizationStrategy
    .getOrElse(_scoringOptimizationStrategyClassifier)
  val _fillConfig: FillConfig = config.fillConfig.getOrElse(_fillConfigDefaults)
  val _outlierConfig: OutlierConfig = config.outlierConfig.getOrElse(_outlierConfigDefaults)
  val _pearsonConfig: PearsonConfig = config.pearsonConfig.getOrElse(_pearsonConfigDefaults)
  val _covarianceConfig: CovarianceConfig = config.covarianceConfig.getOrElse(_covarianceConfigDefaults)

  val _modelParams = ModelingConfig(
    labelCol = config.labelCol,
    featuresCol = config.featuresCol,
    numericBoundaries = config.modelType match {
      case "RandomForest" => config.numericBoundaries.getOrElse(_rfDefaultNumBoundaries)
      case "MLPC" => config.numericBoundaries.getOrElse(_mlpcDefaultNumBoundaries)
    },
    stringBoundaries = config.modelType match {
      case "RandomForest" => config.stringBoundaries.getOrElse(_rfDefaultStringBoundaries)
      case "MLPC" => config.stringBoundaries.getOrElse(_mlpcDefaultStringBoundaries)
    },
    scoringMetric = config.modelType match {
      case "RandomForest" => config.scoringMetric.getOrElse(_scoringDefaultClassifier)
      case "MLPC" => config.scoringMetric.getOrElse(_scoringDefaultClassifier)
    }
  )

  def getModelConfig: ModelingConfig = _modelParams


  def dataPrep():(DataFrame, Array[String], String) = {

    // Fill na values
    val dataSanitizer = new DataSanitizer(config.df)
      .setLabelCol(config.labelCol)
      .setFeatureCol(config.featuresCol)
      .setCharacterFillStat(_fillConfig.characterFillStat)
      .setNumericFillStat(_fillConfig.numericFillStat)
      .setModelSelectionDistinctThreshold(_fillConfig.modelSelectionDistinctThreshold)

    val (filledData, modelType) = if (config.naFillFlag) {
      dataSanitizer.generateCleanData()
    } else {
      (config.df, dataSanitizer.decideModel())
    }

    // Variance Filtering
    val varianceFiltering = new VarianceFiltering(filledData)
      .setLabelCol(config.labelCol)

    val varianceFilteredData = if (config.varianceFilterFlag) varianceFiltering.filterZeroVariance() else filledData


    // Outlier Filtering
    val outlierFiltering = new OutlierFiltering(varianceFilteredData)
      .setLabelCol(config.labelCol)
      .setFilterBounds(_outlierConfig.filterBounds)
      .setLowerFilterNTile(_outlierConfig.lowerFilterNTile)
      .setUpperFilterNTile(_outlierConfig.upperFilterNTile)
      .setFilterPrecision(_outlierConfig.filterPrecision)
      .setContinuousDataThreshold(_outlierConfig.continuousDataThreshold)

    val (outlierCleanedData, removedData) = if (config.outlierFilterFlag) {
      outlierFiltering.filterContinuousOutliers(_outlierConfig.fieldsToIgnore)
    } else {
      (filledData, spark.createDataFrame(sc.emptyRDD[Row], filledData.schema))
    }

    // Construct the Featurized Data Pipeline
    val (preFilteredData, preFilteredFields) = new FeaturePipeline(outlierCleanedData).makeFeaturePipeline()

    // Covariance Filtering
    val covarianceFiltering = new FeatureCorrelationDetection(preFilteredData, preFilteredFields)

    val (postFilteredData, postFilteredFields) = if (config.covarianceFilteringFlag) {
      new FeaturePipeline(covarianceFiltering.filterFeatureCorrelation()).makeFeaturePipeline()
    } else {
      (preFilteredData, preFilteredFields)
    }

    // Pearson Filtering
    val pearsonFiltering = new PearsonFiltering(postFilteredData, postFilteredFields)
      .setLabelCol(config.labelCol)
      .setFeaturesCol(config.featuresCol)
      .setFilterStatistic(_pearsonConfig.filterStatistic)
      .setFilterDirection(_pearsonConfig.filterDirection)
      .setFilterManualValue(_pearsonConfig.filterManualValue)
      .setFilterMode(_pearsonConfig.filterMode)
      .setAutoFilterNTile(_pearsonConfig.autoFilterNTile)

    // Final Featurization and Field Listing Generation (method's output)
    val (outputData, fieldListing) = if (config.pearsonFilteringFlag) {
      new FeaturePipeline(pearsonFiltering.filterFields()).makeFeaturePipeline()
    } else {
      (postFilteredData, postFilteredFields)
    }

    (outputData, fieldListing, modelType)

  }

}
