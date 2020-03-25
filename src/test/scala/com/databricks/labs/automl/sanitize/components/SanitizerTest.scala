package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.sanitize.DataSanitizer
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}
import org.apache.spark.sql.DataFrame

class SanitizerTest extends AbstractUnitSpec {

  final private val ROW_COUNT = 1000
  final private val CATEGORICAL_COLUMNS = Array("d")
  final private val NUMERIC_COLUMNS = Array("a", "b", "c", "e")
  final private val BOOLEAN_COLUMNS = Array("f")
  final private val LABEL_COL = "label"

  def generateData(modelType: String): DataFrame = {

    DiscreteTestDataGenerator.generateSanitizerData(ROW_COUNT, modelType)

  }

  it should "detect categorical columns" in {

    val result = new DataSanitizer(generateData("classifier"))
      .setLabelCol(LABEL_COL)
      .generateCleanData()

    assert(
      CATEGORICAL_COLUMNS.forall(
        result._2.categoricalColumns.keys.toArray.contains
      ),
      "found categorical columns"
    )

  }

  it should "detect numeric columns" in {

    val result = new DataSanitizer(generateData("classifier"))
      .setLabelCol(LABEL_COL)
      .generateCleanData()

    assert(
      NUMERIC_COLUMNS.forall(result._2.numericColumns.keys.toArray.contains),
      "found numeric columns"
    )

  }

  it should "detect boolean columns" in {

    val result = new DataSanitizer(generateData("classifier"))
      .setLabelCol(LABEL_COL)
      .generateCleanData()

    assert(
      BOOLEAN_COLUMNS.forall(result._2.booleanColumns.keys.toArray.contains),
      "found boolean columns"
    )

  }

  it should "detect a classifier model type" in {

    val result = new DataSanitizer(generateData("classifier"))
      .setLabelCol(LABEL_COL)
      .generateCleanData()

    assert(
      result._3 == "classifier",
      "detects classifier type based on label data"
    )

  }

  it should "detect a regressor model type" in {

    val result = new DataSanitizer(generateData("regressor"))
      .setLabelCol(LABEL_COL)
      .generateCleanData()

    assert(
      result._3 == "regressor",
      "detects regressor type based on label data"
    )

  }

}
