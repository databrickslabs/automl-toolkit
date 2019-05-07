package com.databricks.labs.automl.model

import com.databricks.labs.automl.executor.DataPrep
import com.databricks.labs.automl.params.LogisticRegressionModelsWithResults
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class LogisticRegressionTunerTest extends AbstractUnitSpec {

  "LogisticRegressionTuner" should "throw IllegalArgumentException for passing invalid params" in {
    a [IllegalArgumentException] should be thrownBy {
      new LogisticRegressionTuner(null).evolveBest()
    }
  }

  it should "should throw IllegalArgumentException for passing invalid dataset" in {
    a [IllegalArgumentException] should be thrownBy {
      new LogisticRegressionTuner(AutomationUnitTestsUtil.sparkSession.emptyDataFrame).evolveBest()
    }
  }

  it should "should return valid LogisticRegressionModelsWithResults" in {
    val adultDataset = AutomationUnitTestsUtil.getAdultDf()
    val logisticRegressionModelsWithResults: LogisticRegressionModelsWithResults =  new LogisticRegressionTuner(
      new DataPrep(adultDataset).prepData().data)
      .setFirstGenerationGenePool(5)
      .setNumberOfParentsToRetain(2)
      .setNumberOfMutationsPerGeneration(2)
      .evolveBest()
    assert(logisticRegressionModelsWithResults != null, "logisticRegressionModelsWithResults should not have been null")
    assert(logisticRegressionModelsWithResults.evalMetrics != null, "evalMetrics should not have been null")
    assert(logisticRegressionModelsWithResults.model != null, "model should not have been null")
    assert(logisticRegressionModelsWithResults.modelHyperParams != null, "modelHyperParams should not have been null")
  }
}
