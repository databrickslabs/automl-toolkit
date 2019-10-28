package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.AutomationUnitTestsUtil.convertCsvToDf
import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil, PipelineTestUtils}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions.{col, trim, when}

class FeatureEngineeringPipelineContextTest extends AbstractUnitSpec {

//  "FeatureEngineeringPipelineContextTest"
  ignore should "correctly generate feature engineered dataset" in {
    val testVars = PipelineTestUtils.getTestVars()
    // Generate config
    val overrides = Map(
      "labelCol" -> "label", "mlFlowLoggingFlag" -> false,
      "scalingFlag" -> true, "oneHotEncodeFlag" -> true,
      "numericBoundaries" -> Map(
        "numTrees" -> Tuple2(50.0, 1000.0),
        "maxBins" -> Tuple2(10.0, 100.0),
        "maxDepth" -> Tuple2(2.0, 20.0),
        "minInfoGain" -> Tuple2(0.0, 0.075),
        "subSamplingRate" -> Tuple2(0.5, 1.0))
      )
    val randomForestConfig = ConfigurationGenerator.generateConfigFromMap("RandomForest", "classifier", overrides)
    randomForestConfig.switchConfig.outlierFilterFlag = true
    randomForestConfig.featureEngineeringConfig.outlierFilterPrecision = 0.05
    randomForestConfig.featureEngineeringConfig.outlierLowerFilterNTile = 0.05
    randomForestConfig.featureEngineeringConfig.outlierUpperFilterNTile = 0.95
    randomForestConfig.tunerConfig.tunerParallelism = 10
    randomForestConfig.tunerConfig.tunerTrainSplitMethod = "kSample"
    randomForestConfig.tunerConfig.tunerKFold = 1
    randomForestConfig.tunerConfig.tunerTrainPortion = 0.70
    randomForestConfig.tunerConfig.tunerFirstGenerationGenePool = 5
    randomForestConfig.tunerConfig.tunerNumberOfGenerations = 2
    randomForestConfig.tunerConfig.tunerNumberOfParentsToRetain = 2
    randomForestConfig.tunerConfig.tunerNumberOfMutationsPerGeneration = 2
    randomForestConfig.tunerConfig.tunerGeneticMixing = 0.8
    randomForestConfig.tunerConfig.tunerGenerationalMutationStrategy = "fixed"
    randomForestConfig.tunerConfig.tunerEvolutionStrategy = "batch"
    val featuresEngPipelineModel = FeatureEngineeringPipelineContext
      .generatePipelineModel(
        testVars.df,
        ConfigurationGenerator.generateMainConfig(randomForestConfig)
      )
      .pipelineModel
    val pipelineModel =
      PipelineTestUtils.saveAndLoadPipelineModel(featuresEngPipelineModel, testVars.df, "full-feature-eng-pipeline")
    assert(
      pipelineModel.transform(testVars.df)
      .count() == 99, "Total row count shouldn't have changed")
  }

  //
  "FeatureEngineeringPipelineContextTest" should "run train, save/load pipeline and predict" in {
    val testVars = PipelineTestUtils.getTestVars()
    val overrides = Map(
      "labelCol" -> "label",
      "mlFlowLoggingFlag" -> false,
      "scalingFlag" -> true,
      "oneHotEncodeFlag" -> true,
      "numericBoundaries" -> Map(
        "numTrees" -> Tuple2(50.0, 100.0),
        "maxBins" -> Tuple2(10.0, 20.0),
        "maxDepth" -> Tuple2(2.0, 5.0),
        "minInfoGain" -> Tuple2(0.0, 0.03),
        "subSamplingRate" -> Tuple2(0.5, 1.0)),
      "tunerParallelism" -> 10,
      "outlierFilterFlag" -> false,
      "outlierFilterPrecision" -> 0.05,
      "outlierLowerFilterNTile" -> 0.05,
      "outlierUpperFilterNTile" -> 0.95,
      "tunerTrainSplitMethod" -> "kSample",
      "tunerKFold" -> 1,
      "tunerTrainPortion" -> 0.70,
      "tunerFirstGenerationGenePool" -> 5,
      "tunerNumberOfGenerations" -> 2,
      "tunerNumberOfParentsToRetain" -> 1,
      "tunerNumberOfMutationsPerGeneration" -> 1,
      "tunerGeneticMixing" -> 0.8,
      "tunerGenerationalMutationStrategy" -> "fixed",
      "tunerEvolutionStrategy" -> "batch",
      "pipelineDebugFlag" -> true
    )
    val randomForestConfig = ConfigurationGenerator
      .generateConfigFromMap("RandomForest", "classifier", overrides)
    val runner = FamilyRunner(testVars.df, Array(randomForestConfig)).executeWithPipeline()
    val pipelineModel = runner.bestPipelineModel("RandomForest")
    val bcPipelineModel = testVars.df.sparkSession.sparkContext.broadcast(pipelineModel)

    val predictDf = bcPipelineModel.value.transform(testVars.df.drop("label"))
    assert(predictDf.count() == testVars.df.count(),
    "Inference df count should have matched the input dataset")
    assert(testVars.df.columns.filterNot("label".equals(_)).forall(item => predictDf.columns.contains(item)),
    "All original columns must be present in the predict dataset")
    // Test write and load of full inference pipeline
    val pipelineSavePath = AutomationUnitTestsUtil.getProjectDir() + "/target/pipeline-tests/infer-final-pipeline"
    pipelineModel.write.overwrite().save(pipelineSavePath)
    PipelineModel.load(pipelineSavePath).transform(testVars.df.drop("label")).show(100)
  }

  ignore should "run train pipeline" in {
    val overrides = Map(
      "labelCol" -> "label",
      "mlFlowLoggingFlag" -> false,
      "scalingFlag" -> true,
      "oneHotEncodeFlag" -> true,
      "numericBoundaries" -> Map(
        "numTrees" -> Tuple2(50.0, 100.0),
        "maxBins" -> Tuple2(10.0, 20.0),
        "maxDepth" -> Tuple2(2.0, 5.0),
        "minInfoGain" -> Tuple2(0.0, 0.03),
        "subSamplingRate" -> Tuple2(0.5, 1.0)),
      "tunerParallelism" -> 10,
      "outlierFilterFlag" -> true,
      "outlierFilterPrecision" -> 0.05,
      "outlierLowerFilterNTile" -> 0.05,
      "outlierUpperFilterNTile" -> 0.95,
      "tunerTrainSplitMethod" -> "kSample",
      "tunerKFold" -> 1,
      "tunerTrainPortion" -> 0.70,
      "tunerFirstGenerationGenePool" -> 5,
      "tunerNumberOfGenerations" -> 2,
      "tunerNumberOfParentsToRetain" -> 1,
      "tunerNumberOfMutationsPerGeneration" -> 1,
      "tunerGeneticMixing" -> 0.8,
      "tunerGenerationalMutationStrategy" -> "fixed",
      "tunerEvolutionStrategy" -> "batch",
      "pipelineDebugFlag" -> false
    )

    val adultDf = convertCsvToDf("/adult_data.csv")
    var adultDfCleaned = adultDf
    for ( colName <- adultDf.columns) {
      adultDfCleaned = adultDfCleaned
        .withColumn(colName.split("\\s+").mkString+"_trimmed", trim(col(colName)))
        .drop(colName)
    }
    adultDfCleaned = adultDfCleaned.withColumnRenamed("class_trimmed", "label")


    val randomForestConfig = ConfigurationGenerator
      .generateConfigFromMap("RandomForest", "classifier", overrides)
    val runner = FamilyRunner(adultDfCleaned, Array(randomForestConfig)).executeWithPipeline()
    val predictDf = runner.bestPipelineModel("RandomForest").transform(adultDfCleaned.drop("label"))

    // Test write and load of full inference pipeline
    val pipelineSavePath = AutomationUnitTestsUtil.getProjectDir() + "/target/pipeline-tests/infer-final-pipeline-lab"
    runner.bestPipelineModel("RandomForest").write.overwrite().save(pipelineSavePath)
    PipelineModel.load(pipelineSavePath).transform(adultDfCleaned.drop("label")).show(100)
  }
}
