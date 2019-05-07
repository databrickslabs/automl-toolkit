package com.databricks.labs.automl

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.{Partitioner, SparkContext}
import com.databricks.labs.automl.inference.InferencePipeline

object go extends App {

  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local[8]")
    .appName("Databricks Automated ML")
    .getOrCreate()

  lazy val sc: SparkContext = SparkContext.getOrCreate()

  val train = spark.read
    .option("header", true)
    .option("inferSchema", true)
    .option("nullValue", "NA")
    .csv("/tmp/house_prices/train.csv")
    .cache
  val test = spark.read
    .option("header", true)
    .option("inferSchema", true)
    .option("nullValue", "NA")
    .csv("/tmp/house_prices/test.csv").cache
  train.count
  test.count

  val RUNVERSION = 8
  val modelingType = "LinearRegression"
  val labelColumn = "SalePrice"
  val runExperiment = s"housePrices_$RUNVERSION"
  val projectName = "housePrices"

  def doTrain(trainDF: DataFrame): Unit = {
    val fullConfig = new AutomationRunner(trainDF)
      .setModelingFamily(modelingType)
      .setLabelCol(labelColumn)
      .setFeaturesCol("features")
      .setStringBoundaries(Map("loss" -> List("squaredError")))
      .naFillOn()
      .setModelSelectionDistinctThreshold(20)
      .varianceFilterOn()
      .outlierFilterOff()
      .pearsonFilterOff()
      .covarianceFilterOff()
      .oneHotEncodingOff()
      .scalingOn()
      .autoStoppingOff()
      .mlFlowLoggingOn()
      .mlFlowLogArtifactsOff()
      .setMlFlowTrackingURI("http://localhost:5000")
      .setMlFlowExperimentName(s"danTest")
      .setMlFlowModelSaveDirectory(s"/tmp/tomes/ml/automl/danTest/models/")
      .setInferenceConfigSaveLocation(s"/tmp/tomes/ml/automl/danTest/inference/$runExperiment")
      .setFilterPrecision(0.9)
      .setParallelism(8)
      .setKFold(1)
      .setTrainPortion(0.70)
      .setTrainSplitMethod("random")
      .setFirstGenerationGenePool(8)
      .setNumberOfGenerations(4)
      .setNumberOfParentsToRetain(2)
      .setNumberOfMutationsPerGeneration(4)
      .setGeneticMixing(0.8)
      .setGenerationalMutationStrategy("fixed")
      .setScoringMetric("r2")
      .setFeatureImportanceCutoffType("count")
      .setFeatureImportanceCutoffValue(15.0)
      .setEvolutionStrategy("batch")
      .setFirstGenerationMode("random")
      .setFirstGenerationPermutationCount(20)
      .setFirstGenerationIndexMixingMode("random")
      .setFirstGenerationArraySeed(42L)
      .hyperSpaceInferenceOn()
      .setHyperSpaceInferenceCount(400000)
      .setHyperSpaceModelType("LinearRegression")
      .setHyperSpaceModelCount(4)

    val resultData = fullConfig.runWithConfusionReport()
    resultData.confusionData.show()
  }

  def infer(inferDataFrameLocation: String, inferDF: DataFrame): Unit = {
    val inferenceConfig = new InferencePipeline(inferDF)
      .runInferenceFromStoredDataFrame(inferDataFrameLocation)

    inferenceConfig.show()
  }

//  doTrain(train)
  infer("/tmp/tomes/ml/automl/danTest/inference/housePrices_7/_best/f992fe66793b473cba4de9886588d34a_best",
    test)
}
