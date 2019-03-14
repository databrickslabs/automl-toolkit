package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.executor.DataPrep
import com.databricks.spark.automatedml.params.GBTModelsWithResults
import com.databricks.spark.automatedml.{AbstractUnitSpec, AutomationUnitTestsUtil}

class GBTreesTunerTest extends AbstractUnitSpec {

  "GBTreesTuner" should "throw UnsupportedOperationException for passing invalid params" in {
    a [UnsupportedOperationException] should be thrownBy {
      new GBTreesTuner(null, null).evolveBest()
    }
  }

  it should "should throw UnsupportedOperationException for passing invalid modelSelection" in {
    a [UnsupportedOperationException] should be thrownBy {
      new GBTreesTuner(AutomationUnitTestsUtil.getAdultDf(), "err").evolveBest()
    }
  }

  it should "should return valid GBTModelsWithResults" in {
    val adultDataset = AutomationUnitTestsUtil.getAdultDf()
    val gbtModelsWithResults: GBTModelsWithResults =  new GBTreesTuner(
      new DataPrep(adultDataset).prepData().data,
      "regressor")
      .setFirstGenerationGenePool(5)
      .setNumberOfParentsToRetain(2)
      .setNumberOfMutationsPerGeneration(2)
      .evolveBest()
    assert(gbtModelsWithResults != null, "gbtModelsWithResults should not have been null")
    assert(gbtModelsWithResults.evalMetrics != null, "evalMetrics should not have been null")
    assert(gbtModelsWithResults.model != null, "model should not have been null")
    assert(gbtModelsWithResults.modelHyperParams != null, "modelHyperParams should not have been null")
  }

}
