package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.sanitize.FeatureCorrelationDetection
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}

class FeatureCorrelationTest extends AbstractUnitSpec {

  private final val LABEL_COL = "label"
  private final val PARALLELISM = 4

  final val (data, features) =
    DiscreteTestDataGenerator.generateFeatureCorrelationData(1000)

  private def generateFeatureCorrelation(
    high: Double,
    low: Double
  ): FeatureCorrelationDetection = {

    new FeatureCorrelationDetection(data, features)
      .setLabelCol(LABEL_COL)
      .setParallelism(PARALLELISM)
      .setCorrelationCutoffLow(low)
      .setCorrelationCutoffHigh(high)

  }

  it should "execute appropriate filtering with max and min values" in {

    val INTENDED_FIELDS =
      Array("a1", "c2", "d1", "d2", "label", "automl_internal_id")

    val filteredData =
      generateFeatureCorrelation(1.0, -1.0).filterFeatureCorrelation()

    val schemaNames = filteredData.schema.names

    assert(
      INTENDED_FIELDS.forall(schemaNames.contains),
      "appropriate fields have been filtered"
    )

  }
}
// TODO: FINISH THE TESTS
