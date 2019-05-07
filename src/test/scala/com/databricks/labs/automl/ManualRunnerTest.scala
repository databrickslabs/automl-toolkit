package com.databricks.labs.automl

import com.databricks.labs.automl.executor.DataPrep
import com.databricks.labs.automl.params.{ConfusionOutput, DataGeneration}

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

  it should "execute ManualRunner without any exceptions" in {
    val adultDataset = AutomationUnitTestsUtil.getAdultDf()
    val confusionOutput: ConfusionOutput = new ManualRunner(new DataPrep(adultDataset).prepData())
      .setScoringMetric("areaUnderROC")
      .setNumberOfGenerations(2)
      .setFirstGenerationGenePool(5)
      .mlFlowLoggingOff()
      .mlFlowLogArtifactsOff()
      .setInferenceConfigSaveLocation(AutomationUnitTestsUtil.getSerializablesToTmpLocation())
      .runWithConfusionReport()

    assert(confusionOutput != null, "confusionOutput should not have been null")
    assert(confusionOutput.confusionData != null, "confusion data should not have been null")
    assert(confusionOutput.predictionData != null, "prediction data should not have been null")
    assert(confusionOutput.predictionData.count() == adultDataset.count(), "prediction dataset count should have match original dataset's count")
  }



}
