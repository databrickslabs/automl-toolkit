package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.utils.data.CategoricalHandler
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}
import org.apache.spark.sql.DataFrame

class CardinalityCheckTest extends AbstractUnitSpec {

  def dataGeneration(rows: Int): (DataFrame, Array[String]) = {
    DiscreteTestDataGenerator.generateCardinalityFilteringData(rows)
  }

  def cardinalityCalculation(rows: Int,
                             mode: String,
                             limit: Int): Array[String] = {

    val (data, fields) = dataGeneration(rows)

    new CategoricalHandler(data, mode)
      .setCardinalityType("exact")
      .setPrecision(0.0)
      .validateCategoricalFields(fields.toList, limit)

  }

  it should "silently filter out a single column" in {

    val ROWS = 1000
    val MODE = "silent"
    val LIMIT = 50
    val EXPECTED_RETAINED_FIELDS = Array("b", "c")

    val retainFields = cardinalityCalculation(ROWS, MODE, LIMIT)

    assert(
      EXPECTED_RETAINED_FIELDS.forall(retainFields.contains),
      "Filtered correct elements"
    )
  }

  it should "silently filter out all columns" in {
    val ROWS = 1000
    val MODE = "silent"
    val LIMIT = 2
    val EXPECTED_RETAINED_FIELDS = Array.empty[String]

    val retainFields = cardinalityCalculation(ROWS, MODE, LIMIT)

    assert(
      EXPECTED_RETAINED_FIELDS.forall(retainFields.contains),
      "Filtered out correct elements"
    )

  }

  it should "throw an assertion error in warn mode" in {

    val ROWS = 1000
    val MODE = "warn"
    val LIMIT = 20
    val EXPECTED_RETAINED_FIELDS = Array("c")

    intercept[AssertionError] {
      cardinalityCalculation(ROWS, MODE, LIMIT)
    }

  }

  it should "filter nothing" in {

    val ROWS = 1000
    val MODE = "silent"
    val LIMIT = 500
    val EXPECTED_RETAINED_FIELDS = Array("b", "c", "d")

    val retainFields = cardinalityCalculation(ROWS, MODE, LIMIT)

    assert(
      EXPECTED_RETAINED_FIELDS.forall(retainFields.contains),
      "Filtered out correct elements"
    )

  }

}
