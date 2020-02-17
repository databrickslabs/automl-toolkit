package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.sanitize.VarianceFiltering
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}
import org.apache.spark.sql.DataFrame

class VarianceFilteringTest extends AbstractUnitSpec {

  private final val LABEL_COL = "label"
  private final val FEATURES_COL = "features"
  private final val DATETIME_CONVERSION_TYPE = "split"
  private final val PARALLELISM = 1

  def setupVarianceTest(
    rowCount: Int,
    fieldsToIgnore: Array[String] = Array.empty[String]
  ): (DataFrame, Array[String]) = {

    val data = DiscreteTestDataGenerator.generateVarianceFilteringData(rowCount)

    new VarianceFiltering(data)
      .setLabelCol(LABEL_COL)
      .setFeatureCol(FEATURES_COL)
      .setDateTimeConversionType(DATETIME_CONVERSION_TYPE)
      .setParallelism(PARALLELISM)
      .filterZeroVariance(fieldsToIgnore)

  }

  it should "correctly filter out zero variance columns with no exclusions" in {

    val FILTER_COLUMNS = Array("c", "d")

    val result = setupVarianceTest(200)

    val dfSchemaNames = result._1.schema.names

    assert(
      result._2.sameElements(FILTER_COLUMNS),
      "detect the correct columns to remove"
    )

    assert(
      FILTER_COLUMNS.forall(dfSchemaNames.contains) === false,
      "appropriate columns have been dropped"
    )

  }

  it should "correctly filter out zero variance with an exclusion applied" in {

    val FILTER_COLUMNS = Array("c")

    val result = setupVarianceTest(200, Array("d"))

    val dfSchemaNames = result._1.schema.names

    assert(
      result._2.sameElements(FILTER_COLUMNS),
      "detect the correct columns to remove and ignore the correct columns"
    )
    assert(
      FILTER_COLUMNS.forall(dfSchemaNames.contains) === false,
      "appropriate columns have been dropped"
    )

  }

}
