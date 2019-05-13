package com.databricks.labs.automl

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.{Partitioner, SparkContext}
import com.databricks.labs.automl.inference.InferencePipeline

object go_lr extends App {

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
    val fullConfig = new AutomationRunner(trainDF.sample(false,0.02))
      .setModelingFamily(modelingType)
      .setLabelCol(labelColumn)
      .setFeaturesCol("features")
      .setStringBoundaries(Map("loss" -> List("squaredError")))
      .dataPrepCachingOff()
      .naFillOn()
      .setModelSelectionDistinctThreshold(20)
      .varianceFilterOn()
      .outlierFilterOff()
      .pearsonFilterOff()
      .covarianceFilterOff()
      .oneHotEncodingOff()
      .scalingOn()
      .autoStoppingOff()
      .mlFlowLoggingOff()
      .mlFlowLogArtifactsOff()
      .setMlFlowLoggingMode("full")
      .setMlFlowTrackingURI("http://localhost:5000")
      .setMlFlowExperimentName(s"danTest")
      .setMlFlowModelSaveDirectory(s"/tmp/tomes/ml/automl/danTest/models/")
      .setInferenceConfigSaveLocation(s"/tmp/tomes/ml/automl/danTest/inference/$runExperiment")
      .setFilterPrecision(0.9)
      .setParallelism(8)
      .setKFold(1)
      .setTrainPortion(0.70)
      .setTrainSplitMethod("random")
      .setFirstGenerationGenePool(5)
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

    val rfBoundaries = Map(
      "numTrees" -> Tuple2(50.0, 1000.0),
      "maxBins" -> Tuple2(10.0, 100.0),
      "maxDepth" -> Tuple2(2.0, 20.0),
      "minInfoGain" -> Tuple2(0.0, 0.075),
      "subSamplingRate" -> Tuple2(0.5, 1.0)
    )

    val fiGenConfig = fullConfig.getGeneticConfig.copy(
      firstGenerationGenePool = 5,
      kFold = 1,
      numberOfGenerations = 5,
      numberOfMutationsPerGeneration = 4,
      initialGenerationMode = "random",
      trainSplitMethod = "random",
      hyperSpaceInference = true,
      hyperSpaceInferenceCount = 200000,
      hyperSpaceModelType = "RandomForest",
      hyperSpaceModelCount = 4
    )

    val fiFeatCfg = fullConfig.getFeatConfig.copy(numericBoundaries = rfBoundaries,
      geneticConfig = fiGenConfig
//      labelCol = labelColumn,
//      dataPrepCachingFlag = false,
//      scoringMetric = "r2"
    )

    val results = fullConfig
      .setFeatConfig(fiFeatCfg)
      .runFeatureCullingWithPrediction()

    results.featureImportances.show(false)
    results.predictionData.show(false)
  }

  def infer(inferDataFrameLocation: String, inferDF: DataFrame): Unit = {
    val inferenceConfig = new InferencePipeline(inferDF)
      .runInferenceFromStoredDataFrame(inferDataFrameLocation)

    inferenceConfig.show()
  }

  doTrain(train)
  //  infer("/tmp/tomes/ml/automl/danTest/inference/housePrices_7/_best/f992fe66793b473cba4de9886588d34a_best",
  //    test)
}
