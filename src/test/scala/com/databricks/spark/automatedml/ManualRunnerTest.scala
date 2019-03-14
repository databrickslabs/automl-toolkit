package com.databricks.spark.automatedml

import com.databricks.spark.automatedml.params.DataGeneration
import com.databricks.spark.automatedml.utils.ModelType

class ManualRunnerTest extends AbstractUnitSpec {


  "ManualRunner " should "throw NullPointerException if it is instantiated with null constructor" in {
    a [NullPointerException] should be thrownBy {
      new ManualRunner(null)
    }
  }

  it should "throw NullPointerException if it is instantiated with wrong object state " in {
    a[NullPointerException] should be thrownBy {
      new ManualRunner(DataGeneration(null, null, null)).run()
    }
  }

  it should "return feature importance without any exceptions " in {

    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()

    val manualRunner = new ManualRunner(
      DataGeneration(adultDfwithLabel, adultDfwithLabel.columns, ModelType.CLASSIFIER.toString))
        .mlFlowLoggingOff()
        .mlFlowLogArtifactsOff()
        .oneHotEncodingOn()
        .setEvolutionStrategy("continuous")
        .naFillOn()

    manualRunner.runWithConfusionReport()

    val mo = manualRunner.run()

    assert(manualRunner != null)

    assert(manualRunner.exploreFeatureImportances() != null)

  }



}
