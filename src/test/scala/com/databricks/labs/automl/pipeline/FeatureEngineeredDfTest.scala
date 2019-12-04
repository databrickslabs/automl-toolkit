package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}

class FeatureEngineeredDfTest extends AbstractUnitSpec {
  "It" should "return feature engineered df" in {
    val testVars = PipelineTestUtils.getTestVars()
    val overrides = Map(
      "labelCol" -> "label", "mlFlowLoggingFlag" -> false,
      "scalingFlag" -> true, "oneHotEncodeFlag" -> true,
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
      "tunerKFold" -> 1,
      "tunerTrainPortion" -> 0.70,
      "tunerFirstGenerationGenePool" -> 5,
      "tunerNumberOfGenerations" -> 2,
      "tunerNumberOfParentsToRetain" -> 1,
      "tunerNumberOfMutationsPerGeneration" -> 1,
      "tunerGeneticMixing" -> 0.8,
      "tunerGenerationalMutationStrategy" -> "fixed",
      "tunerEvolutionStrategy" -> "batch",
      "pipelineDebugFlag" -> true,
      "mlFlowLoggingFlag" -> false
    )
    val randomForestConfig = ConfigurationGenerator
      .generateConfigFromMap("RandomForest", "classifier", overrides)
    val featEngPipe = FamilyRunner(testVars.df, Array(randomForestConfig)).generateFeatureEngineeredPipeline()

    featEngPipe("RandomForest").transform(testVars.df).show(100)
  }

}
