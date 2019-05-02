package com.databricks.labs.automl.model

import com.databricks.labs.automl.executor.DataPrep
import com.databricks.labs.automl.params.XGBoostModelsWithResults
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class XgBoostTunerTest extends AbstractUnitSpec {


  "XgBoostTuner" should "throw UnsupportedOperationException for passing invalid params" in {
    a [UnsupportedOperationException] should be thrownBy {
      new XGBoostTuner(null, null).evolveBest()
    }
  }

  it should "should throw UnsupportedOperationException for passing invalid modelSelection" in {
    a [UnsupportedOperationException] should be thrownBy {
      new XGBoostTuner(AutomationUnitTestsUtil.getAdultDf(), "err").evolveBest()
    }
  }

  it should "should return valid XGBoostModelsWithResults" in {
    val adultDataset = AutomationUnitTestsUtil.getAdultDf()
    val xGBoostModelsWithResults: XGBoostModelsWithResults =  new XGBoostTuner(
      new DataPrep(adultDataset).prepData().data, "classifier")
      .setFirstGenerationGenePool(5)
      .setNumberOfParentsToRetain(2)
      .setNumberOfMutationsPerGeneration(2)
      .evolveBest()
    assert(xGBoostModelsWithResults != null, "xGBoostModelsWithResults should not have been null")
    assert(xGBoostModelsWithResults.evalMetrics != null, "evalMetrics should not have been null")
    assert(xGBoostModelsWithResults.model != null, "model should not have been null")
    assert(xGBoostModelsWithResults.modelHyperParams != null, "modelHyperParams should not have been null")
  }
}
