package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.sanitize.DataSanitizer
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}
import org.apache.spark.sql.types._

class NAFillTest extends AbstractUnitSpec {

  private final val LABEL_COL = "label"
  private final val FEATURE_COL = "features"
  private final val DISTINCT_THRESHOLD = 10
  private final val PARALLELISM = 4
  private final val FILTER_PRECISION = 0.01
  private final val EXPECTED_MODEL_TYPE = "classifier"

  def generateNAFillConfigTest(
    numFillStat: String,
    catFillStat: String,
    mode: String,
    expectedContinuousFillMap: Map[String, Double],
    expectedCategoricalFillMap: Map[String, String],
    expectedBooleanFillMap: Map[String, Boolean]
  ): Unit = {

    val expectedPreFillSchema = StructType(
      Seq(
        StructField("dblData", DoubleType, nullable = true),
        StructField("fltData", FloatType, nullable = true),
        StructField("intData", IntegerType, nullable = true),
        StructField("ordinalIntData", IntegerType, nullable = true),
        StructField("strData", StringType, nullable = true),
        StructField("boolData", BooleanType, nullable = false),
        StructField("dateData", DateType, nullable = true),
        StructField("label", IntegerType, nullable = false),
        StructField("automl_internal_id", LongType, nullable = false)
      )
    )

    val expectedPostFillSchema = StructType(
      Seq(
        StructField("dblData", DoubleType, nullable = false),
        StructField("fltData", FloatType, nullable = false),
        StructField("intData", IntegerType, nullable = true),
        StructField("ordinalIntData", IntegerType, nullable = true),
        StructField("strData", StringType, nullable = false),
        StructField("boolData", BooleanType, nullable = false),
        StructField("dateData", DateType, nullable = true),
        StructField("label", IntegerType, nullable = false),
        StructField("automl_internal_id", LongType, nullable = false)
      )
    )

    val data = DiscreteTestDataGenerator.generateNAFillData(100, 10)

    val sanitizer = new DataSanitizer(data)
      .setLabelCol(LABEL_COL)
      .setFeatureCol(FEATURE_COL)
      .setModelSelectionDistinctThreshold(DISTINCT_THRESHOLD)
      .setNumericFillStat(numFillStat)
      .setCharacterFillStat(catFillStat)
      .setParallelism(PARALLELISM)
      .setCategoricalNAFillMap(Map.empty[String, String])
      .setCharacterNABlanketFillValue("")
      .setNumericNABlanketFillValue(Double.NaN)
      .setNumericNAFillMap(Map.empty[String, Double])
      .setNAFillMode(mode)
      .setFilterPrecision(FILTER_PRECISION)
      .setFieldsToIgnoreInVector(
        Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
      )

    val (naFilledDF, fillMap, modelType) = sanitizer.generateCleanData()

    // Validate the incoming schema is correct for the test
    assert(
      data.schema == expectedPreFillSchema,
      "for pre naFill schema validation"
    )

    // Validate the post-fill schema
    assert(
      naFilledDF.schema == expectedPostFillSchema,
      "for post naFill schema validation"
    )

    // Make sure the fill for continuous data is correct
    assert(
      fillMap.numericColumns == expectedContinuousFillMap,
      "for numeric fill na values"
    )

    // Make sure the fill for categorical data is correct
    assert(
      fillMap.categoricalColumns == expectedCategoricalFillMap,
      "for categorical fill na values"
    )
    // Make sure the fill for boolean data is correct
    assert(
      fillMap.booleanColumns == expectedBooleanFillMap,
      "for boolean fill na values"
    )

    // Ensure the model type is correct
    assert(modelType == EXPECTED_MODEL_TYPE, "for model type detection")

  }

  it should "correctly fill na's in auto mode, numeric mean, categorical max" in {

    val NUM_FILL_STAT = "mean"
    val CAT_FILL_STAT = "max"

    val expectedContinuousFillMap = Map(
      "dblData" -> 50.0,
      "fltData" -> 49.0,
      "intData" -> 49.111111111111114,
      "ordinalIntData" -> 2.2222222222222223
    )

    val expectedCategoricalFillMap = Map("strData" -> "e")

    val expectedBooleanFillMap = Map("boolData" -> false)

    generateNAFillConfigTest(
      "mean",
      "max",
      "auto",
      expectedContinuousFillMap,
      expectedCategoricalFillMap,
      expectedBooleanFillMap
    )

  }

  it should "correctly fill na's in auto mode, numeric max, categorical min" in {

    val expectedContinuousFillMap = Map(
      "dblData" -> 99.0,
      "fltData" -> 98.0,
      "intData" -> 99.0,
      "ordinalIntData" -> 4.0
    )

    val expectedCategoricalFillMap = Map("strData" -> "a")

    val expectedBooleanFillMap = Map("boolData" -> true)

    generateNAFillConfigTest(
      "max",
      "min",
      "auto",
      expectedContinuousFillMap,
      expectedCategoricalFillMap,
      expectedBooleanFillMap
    )

  }

  it should "correctly fill na's in auto mode, numeric median, categorical min" in {

    val expectedContinuousFillMap = Map(
      "dblData" -> 49.0,
      "fltData" -> 48.0,
      "intData" -> 49.0,
      "ordinalIntData" -> 2.0
    )

    val expectedCategoricalFillMap = Map("strData" -> "a")

    val expectedBooleanFillMap = Map("boolData" -> true)

    generateNAFillConfigTest(
      "median",
      "min",
      "auto",
      expectedContinuousFillMap,
      expectedCategoricalFillMap,
      expectedBooleanFillMap
    )

  }

}
//TODO: test -> "auto", "mapFill", "blanketFillAll", "blanketFillCharOnly", "blanketFillNumOnly"

// TODO: add assertions for no nulls in the columns that should have data filled!!!
