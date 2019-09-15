package com.databricks.labs.automl.model

import com.databricks.labs.automl.executor.DataPrep
import com.databricks.labs.automl.params.{Defaults, LinearRegressionModelsWithResults}
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class LinearRegressionTunerTest extends AbstractUnitSpec {
  "LinearRegressionTuner" should "throw NoSuchElementException for passing invalid params" in {
    a [NullPointerException] should be thrownBy {
      new LinearRegressionTuner(null).evolveBest()
    }
  }

  it should "should throw NoSuchElementException for passing invalid dataset" in {
    a [AssertionError] should be thrownBy {
      new LinearRegressionTuner(AutomationUnitTestsUtil.sparkSession.emptyDataFrame).evolveBest()
    }
  }

  it should "should return valid LinearRegressionModelsWithResults" in {
    val adultDataset = AutomationUnitTestsUtil.convertCsvToDf("/AirQualityUCI.csv")
    val linearRegressionModelsWithResults: LinearRegressionModelsWithResults =  new LinearRegressionTuner(
      new DataPrep(adultDataset).prepData().data)
      .setFirstGenerationGenePool(5)
      .setNumberOfParentsToRetain(2)
      .setNumberOfMutationsPerGeneration(2)
      .setScoringMetric("rmse")
      .setLinearRegressionNumericBoundaries(Map (
        "elasticNetParams" -> Tuple2(0.0, 0.0),
        "maxIter" -> Tuple2(100.0, 10000.0),
        "regParam" -> Tuple2(0.0, 1.0),
        "tolerance" -> Tuple2(1E-9, 1E-5)
      ))
      .setLinearRegressionStringBoundaries(new Defaults{}._linearRegressionDefaultStringBoundaries)
      .evolveBest()
    assert(linearRegressionModelsWithResults != null, "linearRegressionModelsWithResults should not have been null")
    assert(linearRegressionModelsWithResults.evalMetrics != null, "evalMetrics should not have been null")
    assert(linearRegressionModelsWithResults.model != null, "model should not have been null")
    assert(linearRegressionModelsWithResults.modelHyperParams != null, "modelHyperParams should not have been null")
  }
}
