package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.exceptions.FeatureCorrelationException
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

  it should "throw an exception if all fields have been filtered out" in {

    intercept[RuntimeException] {
      generateFeatureCorrelation(-0.1, 0.1).filterFeatureCorrelation()
    }
  }

  it should "throw an exception for improper configuration" in {

    intercept[FeatureCorrelationException] {
      generateFeatureCorrelation(-10, 0.0).filterFeatureCorrelation()
    }

  }

  it should "filter appropriate fields with removing positive correlation" in {

    val INTENDED_FIELDS =
      Array("a1", "c2", "d1", "d2", "label", "automl_internal_id")

    val filteredData =
      generateFeatureCorrelation(0.1, -1.0).filterFeatureCorrelation()

    val schemaNames = filteredData.schema.names
    assert(
      INTENDED_FIELDS.forall(schemaNames.contains),
      "appropriate fields filtered"
    )

  }

  it should "filter appropriate number of fields with removing negative correlation" in {

    val expectedRemainingFieldCount = 6

    val filteredData =
      generateFeatureCorrelation(1.0, -0.1).filterFeatureCorrelation()

    val schemaNames = filteredData.schema.names

    assert(
      schemaNames.length == expectedRemainingFieldCount,
      "appropriate fields filtered"
    )

  }

}
