package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.executor.DataPrep
import com.databricks.spark.automatedml.params.RandomForestModelsWithResults
import com.databricks.spark.automatedml.{AbstractUnitSpec, AutomationUnitTestsUtil}

class RandomForestTunerTest extends AbstractUnitSpec {

  "RandomForestTuner" should "throw UnsupportedOperationException for passing invalid params" in {
    a [UnsupportedOperationException] should be thrownBy {
      new RandomForestTuner(null, null).evolveBest()
    }
  }

  it should "should throw UnsupportedOperationException for passing invalid modelSelection" in {
    a [UnsupportedOperationException] should be thrownBy {
      new RandomForestTuner(AutomationUnitTestsUtil.getAdultDf(), "err").evolveBest()
    }
  }

  it should "should return valid RandomForestModelsWithResults" in {
    val adultDataset = AutomationUnitTestsUtil.getAdultDf()
    val randomForestModelsWithResults: RandomForestModelsWithResults =  new RandomForestTuner(
      new DataPrep(adultDataset).prepData().data, "regressor")
      .setFirstGenerationGenePool(5)
      .setNumberOfParentsToRetain(2)
      .setNumberOfMutationsPerGeneration(2)
      .evolveBest()
    assert(randomForestModelsWithResults != null, "randomForestModelsWithResults should not have been null")
    assert(randomForestModelsWithResults.evalMetrics != null, "evalMetrics should not have been null")
    assert(randomForestModelsWithResults.model != null, "model should not have been null")
    assert(randomForestModelsWithResults.modelHyperParams != null, "modelHyperParams should not have been null")
  }
}
