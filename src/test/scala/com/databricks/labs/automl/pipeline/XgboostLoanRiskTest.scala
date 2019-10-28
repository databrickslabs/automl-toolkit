package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class XgboostLoanRiskTest extends AbstractUnitSpec {

  "it" should "do this" in {
    val loanRiskDf = AutomationUnitTestsUtil.convertCsvToDf("/loan_risk.csv")
    val genericMapOverrides = Map(
      "labelCol" -> "label",
      "scoringMetric" -> "areaUnderROC",
      "oneHotEncodeFlag" -> true,
      "autoStoppingFlag" -> true,
      "tunerAutoStoppingScore" -> 0.91,
      "tunerParallelism" -> 1 * 2,
      "tunerKFold" -> 2,
      "tunerTrainPortion" -> 0.7,
      "tunerTrainSplitMethod" -> "stratified",
      "tunerInitialGenerationMode" -> "permutations",
      "tunerInitialGenerationPermutationCount" -> 8,
      "tunerInitialGenerationIndexMixingMode" -> "linear",
      "tunerInitialGenerationArraySeed" -> 42L,
      "tunerFirstGenerationGenePool" -> 16,
      "tunerNumberOfGenerations" -> 3,
      "tunerNumberOfParentsToRetain" -> 2,
      "tunerNumberOfMutationsPerGeneration" -> 4,
      "tunerGeneticMixing" -> 0.8,
      "tunerGenerationalMutationStrategy" -> "fixed",
      "tunerEvolutionStrategy" -> "batch",
      "tunerHyperSpaceInferenceFlag" -> true,
      "tunerHyperSpaceInferenceCount" -> 400000,
      "tunerHyperSpaceModelType" -> "XGBoost",
      "tunerHyperSpaceModelCount" -> 8,
      "mlFlowLoggingFlag" -> false,
      "mlFlowLogArtifactsFlag" -> false,
      "pipelineDebugFlag" -> true
    )
    val xgBoostConfig = ConfigurationGenerator.generateConfigFromMap("XGBoost", "classifier", genericMapOverrides)
    val familyRunner = FamilyRunner(loanRiskDf, Array(xgBoostConfig)).executeWithPipeline()
    familyRunner.bestPipelineModel(("XGBoost")).transform(loanRiskDf).show(10)
  }

}
