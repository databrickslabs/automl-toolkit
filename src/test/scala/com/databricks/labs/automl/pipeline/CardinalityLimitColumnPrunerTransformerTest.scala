package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.AutomationUnitTestsUtil.getClass
import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil, PipelineTestUtils}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ArrayBuffer

class CardinalityLimitColumnPrunerTransformerTest extends AbstractUnitSpec {

  "CardinalityLimitColumnPrunerTransformerTest" should " should check cardinality" in {
    val testVars = PipelineTestUtils.getTestVars()
    val stages = new ArrayBuffer[PipelineStage]
    val nonFeatureCols =
      Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL, testVars.labelCol)
    stages += PipelineTestUtils
      .addZipRegisterTmpTransformerStage(
        testVars.labelCol,
        testVars.df.columns.filterNot(item => nonFeatureCols.contains(item))
      )
    stages += new CardinalityLimitColumnPrunerTransformer()
      .setLabelColumn(testVars.labelCol)
      .setCardinalityLimit(2)
      .setCardinalityCheckMode("silent")
      .setCardinalityType("exact")
      .setCardinalityPrecision(0.0)
    val pipelineModel = PipelineTestUtils.saveAndLoadPipeline(
      stages.toArray,
      testVars.df,
      "card-limit-pipeline"
    )
    val adultCadDf = pipelineModel.transform(testVars.df)
    assertCardinalityTest(adultCadDf)
    adultCadDf.show(10)
  }

  private def assertCardinalityTest(adultCadDf: DataFrame): Unit = {
    assert(
      adultCadDf.columns
        .exists(item => Array("sex_trimmed", "label").contains(item)),
      "CardinalityLimitColumnPrunerTransformer should have retained columns with a defined cardinality"
    )
  }


  "Categorical fields" should "work consistently for trees with cardinality transformer" in {

    val wineDf = AutomationUnitTestsUtil.convertCsvToDf("/ml_wine.csv")

    val configurationOverrides = Map(
      "labelCol" -> "class",
      "dataPrepCachingFlag" -> true,
      "tunerParallelism" -> 4,
      "tunerKFold" -> 1,
      "tunerTrainSplitMethod" -> "kSample",
      "featureInteractionFlag" -> false,
      "scoringMetric" -> "f1",
      "featureInteractionRetentionMode" -> "all",
      "tunerNumberOfGenerations" -> 3, //10
      "tunerNumberOfMutationsPerGeneration" -> 3, //50
      "tunerInitialGenerationMode" -> "permutations",
      "tunerInitialGenerationPermutationCount" -> 8,
      "tunerFirstGenerationGenePool" -> 8,
      "pipelineDebugFlag" -> true,
      "fillConfigCardinalityLimit" -> 10,
      "mlFlowLoggingFlag" -> false
    )

    val configsPayload = Array(ConfigurationGenerator.generateConfigFromMap("LogisticRegression", "classifier", configurationOverrides))

    val runnerBinaryPipeline = FamilyRunner(wineDf.repartition(4), configsPayload).executeWithPipeline()

    PipelineTestUtils.saveAndLoadPipelineModel(
      runnerBinaryPipeline.bestPipelineModel("LogisticRegression"),
      wineDf,
    "ml_wine_pipeline")
  }

}
