package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.sanitize.PearsonFiltering
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}
import org.apache.spark.sql.DataFrame

class PearsonFilteringTest extends AbstractUnitSpec {

  private final val TARGET_ROWS = 500
  private final val DEBUG_COUNT = 10
  private final val LABEL_COL = "label"
  private final val FEATURES_COL = "features"

  def instantiatePearson(data: DataFrame,
                         features: Array[String],
                         modelType: String): PearsonFiltering = {

    new PearsonFiltering(data, features, modelType)
      .setLabelCol(LABEL_COL)
      .setFeaturesCol(FEATURES_COL)

  }

  it should "filter intended columns by using pvalue filtering with auto thresholding" in {

    val INTENDED_FILTER = Array("positiveCorr1", "positiveCorr2")

    val (data, fields) =
      DiscreteTestDataGenerator.generatePearsonFilteringData(TARGET_ROWS)

    val filtered = instantiatePearson(data, fields, "classifier")
      .setFilterStatistic("pvalue")
      .setFilterDirection("greater")
      .filterFields()

    assert(
      INTENDED_FILTER.forall(filtered.schema.names.contains) === false,
      "appropriate columns have been dropped"
    )

    filtered.show(DEBUG_COUNT)

  }

  it should "filter columns using pvalue except for ignored column" in {

    val INTENDED_FILTER = Array("positiveCorr1")

    val (data, fields) =
      DiscreteTestDataGenerator.generatePearsonFilteringData(TARGET_ROWS)

    val filtered = instantiatePearson(data, fields, "classifier")
      .filterFields(Array("positiveCorr2"))

    assert(
      INTENDED_FILTER.forall(filtered.schema.names.contains) === false,
      "appropriate columns have been dropped and excluded fields remain"
    )
    filtered.show(DEBUG_COUNT)

  }

  it should "filter correctly for a regression problem" in {

    val INTENDED_FILTER =
      Array("positiveCorr1", "positiveCorr2", "positiveCorr3")

    val (data, fields) =
      DiscreteTestDataGenerator.generatePearsonRegressionFilteringData(
        TARGET_ROWS
      )

    val filtered = instantiatePearson(data, fields, "regressor")
      .setFilterMode("auto")
      .setAutoFilterNTile(0.99)
      .filterFields()

    assert(
      INTENDED_FILTER.forall(filtered.schema.names.contains) === false,
      "appropriate columns have been dropped."
    )

    filtered.show(DEBUG_COUNT)

  }

  it should "filter correctly for regression except for ignored columns" in {

    val INTENDED_FILTER = Array("positiveCorr2")

    val (data, fields) =
      DiscreteTestDataGenerator.generatePearsonRegressionFilteringData(
        TARGET_ROWS
      )

    val filtered = instantiatePearson(data, fields, "regressor")
      .setFilterMode("manual")
      .setFilterManualValue(0.9)
      .filterFields(Array("positiveCorr1", "positiveCorr3"))

    assert(
      INTENDED_FILTER.forall(filtered.schema.names.contains) === false,
      "appropriate columns have been dropped and exclusions have been ignored."
    )
    filtered.show(DEBUG_COUNT)

  }
}
