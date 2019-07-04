package com.databricks.labs.automl

import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig}
import com.databricks.labs.automl.exploration.FeatureImportances
import com.databricks.labs.automl.inference.InferencePipeline
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SparkSession}

object L1 extends App {

  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("Databricks Automated ML")
    .getOrCreate()

  lazy val sc: SparkContext = SparkContext.getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  val sampleSize = 0.005

  //  First Time -- Building the sample
  //  val splitDF = spark.read
  //    .option("header", true)
  //    .option("inferSchema", true)
  //    .option("nullValue", "NA")
  //    .parquet("/Users/danieltomes/data/L1_sample/iss_480")
  //    .randomSplit(Array(70, 30), 400L)
  //
  //  val (train_raw, test_raw) = (splitDF(0), splitDF(1))
  //  val train_t = train_raw.sample(sampleSize).repartition(8)
  //  val test = test_raw.sample(sampleSize).repartition(8)
  //  train_t.write.format("parquet").mode("overwrite").save("/Users/danieltomes/data/L1_sample/iss_480_sample")

  val train = spark.read.format("parquet").load("/home/tomesd/dev/data/L1_Sample/iss_480_sample").cache()
  train.count()

  val RUNVERSION = 3
  val modelingType = "RandomForest"
  val labelColumn = "TARGET"
  val runExperiment = s"adult_income_$RUNVERSION"
  val projectName = "adult"


  val rfBoundaries = Map(
    "numTrees" -> Tuple2(50.0, 1000.0),
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 0.075),
    "subSamplingRate" -> Tuple2(0.5, 1.0)
  )

  val genericMapOverrides = Map(
    "labelCol" -> labelColumn,
    "scoringMetric" -> "areaUnderROC",
    "fieldsToIgnoreInVector" -> Array("COLLECTOR_NUMBER"),
    "dataPrepCaching" -> false,
    "autoStoppingFlag" -> true,
    "tunerAutoStoppingScore" -> 0.9,
    "tunerParallelism" -> 8,
    "tunerKFold" -> 2,
    "tunerTrainPortion" -> 0.7,
    "tunerTrainSplitMethod" -> "random",
    "tunerFirstGenerationGenePool" -> 8,
    "tunerNumberOfGenerations" -> 3,
    "tunerNumberOfParentsToRetain" -> 2,
    "tunerNumberOfMutationsPerGeneration" -> 4,
    "tunerGeneticMixing" -> 0.8,
    "tunerGenerationalMutationStrategy" -> "fixed",
    "tunerEvolutionStrategy" -> "batch",
    "tunerInitialGenerationMode" -> "permutations",
    "tunerInitialGenerationPermutationCount" -> 8,
    "tunerInitialGenerationIndexMixingMode" -> "random",
    "tunerInitialGenerationArraySeed" -> 42L,
    "tunerHyperSpaceInference" -> true,
    "tunerHyperSpaceInferenceCount" -> 400000,
    "tunerHyperSpaceModelType" -> "XGBoost",
    "tunerHyperSpaceModelCount" -> 8
//    "mlFlowLoggingFlag" -> ,
//    "mlFlowLogArtifactsFlag" -> ,
//    "mlFlowTrackingURI" -> ,
//    "mlFlowExperimentName" -> ,
//    "mlFlowAPIToken" -> ,
//    "mlFlowModelSaveDirectory" -> ,
//    "mlFlowLoggingMode" -> ,
//    "mlFlowBestSuffix" ->

  )

  val fiConfig = ConfigurationGenerator.generateConfigFromMap("XGBoost", "classifer", genericMapOverrides)


  val fiMainConfig = ConfigurationGenerator.generateFeatureImportanceConfig(fiConfig)

  val importances = new FeatureImportances(train, fiMainConfig, "count", 20.0)
    .generateFeatureImportances()

  importances.importances.show()

  val importantFeatures = importances.topFields

//
  val conf: InstanceConfig = ConfigurationGenerator.generateDefaultConfig("RandomForest", "classifier")



  new XGBoostClassifier().setNumWorkers(4)

//
//
//  //  Algo Fine Grained Config
//  conf.algorithmConfig.numericBoundaries = rfBoundaries
//
//  // Finalize FI Config
////  val fiMainConfig = ConfigurationGenerator.generateMainConfig(conf)
//  conf.featureEngineeringConfig.scalingStdDevFlag
//
//  // Mutate Config for Model and instantiate new config
//  //  Logging Config
//  conf.loggingConfig.mlFlowLoggingFlag = false
//  conf.loggingConfig.mlFlowLogArtifactsFlag = false
//  //  conf.loggingConfig.mlFlowAPIToken = dbutils.notebook.getContext().apiToken.get
//  //  conf.loggingConfig.mlFlowExperimentName = s"/Users/DTomes@loyalty.com/issuance_churn/tracking/$projectName/$runExperiment"
//  //  conf.loggingConfig.mlFlowModelSaveDirectory = s"dbfs:/tmp/tomes/ml/automl/models/$projectName/"
//  //  conf.loggingConfig.inferenceConfigSaveLocation = s"dbfs:/tmp/tomes/ml/automl/inference/$projectName/"
//  //  conf.loggingConfig.mlFlowBestSuffix = "_best"
//  //  conf.loggingConfig.mlFlowLoggingMode = "bestOnly"
//  //  conf.loggingConfig.mlFlowTrackingURI = "https://loyaltyone-ca.cloud.databricks.com"
//
//  //  Adjust model tuner configs
//  conf.switchConfig.autoStoppingFlag = true
//  conf.tunerConfig.tunerAutoStoppingScore = 0.93
//  conf.tunerConfig.tunerKFold = 2
//  conf.tunerConfig.tunerNumberOfGenerations = 3
//
//  val modelMainConfig = ConfigurationGenerator.generateMainConfig(conf)
//
//  val runner = new AutomationRunner(train)
//    .setFeatConfig(fiMainConfig)
//    .setMainConfig(modelMainConfig)
//    .exploreFeatureImportances()
//
//  runner.data.show()
//  //  runner.featureImportances.show()
//  //  runner.predictionData.show()


}
