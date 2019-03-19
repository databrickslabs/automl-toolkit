package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.executor.DataPrep
import com.databricks.spark.automatedml.params.LogisticRegressionModelsWithResults
import com.databricks.spark.automatedml.{AbstractUnitSpec, AutomationUnitTestsUtil}

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
