package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.inference.NaFillConfig
import com.databricks.labs.automl.sanitize.DataSanitizer
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._

class NAFillTest extends AbstractUnitSpec {

  private final val LABEL_COL = "label"
  private final val FEATURE_COL = "features"
  private final val DISTINCT_THRESHOLD = 10
  private final val PARALLELISM = 4
  private final val FILTER_PRECISION = 0.01
  private final val EXPECTED_MODEL_TYPE = "classifier"
  private final val NA_FILL_ROW_COUNT = 100
  private final val NA_RATE = 10

  private final val STRING_FILL_MAP = Map("strData" -> "zzzz")
  private final val NUMERIC_FILL_MAP = Map(
    "dblData" -> 99999.99,
    "fltData" -> 99999.9f,
    "intData" -> 9999,
    "ordinalIntData" -> 9999
  )
  private final val CHAR_FILL_VALUE = "hodor"
  private final val NUM_FILL_VALUE = -999.0

  def setupNAFillTest(numFillStat: String,
                      catFillStat: String,
                      fillMode: String): DataSanitizer = {

    val data =
      DiscreteTestDataGenerator.generateNAFillData(NA_FILL_ROW_COUNT, NA_RATE)

    new DataSanitizer(data)
      .setLabelCol(LABEL_COL)
      .setFeatureCol(FEATURE_COL)
      .setModelSelectionDistinctThreshold(DISTINCT_THRESHOLD)
      .setNumericFillStat(numFillStat)
      .setCharacterFillStat(catFillStat)
      .setParallelism(PARALLELISM)
      .setCategoricalNAFillMap(STRING_FILL_MAP)
      .setCharacterNABlanketFillValue(CHAR_FILL_VALUE)
      .setNumericNABlanketFillValue(NUM_FILL_VALUE)
      .setNumericNAFillMap(NUMERIC_FILL_MAP)
      .setNAFillMode(fillMode)
      .setFilterPrecision(FILTER_PRECISION)
      .setFieldsToIgnoreInVector(
        Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
      )

  }

  def setupMapFillModes(mode: String): (DataFrame, NaFillConfig, String) = {

    val sanitizer = setupNAFillTest("mean", "max", mode)

    sanitizer.generateCleanData()

  }

  def checkForNulls(df: DataFrame, columnName: String): Unit = {

    assert(
      df.na.drop(Seq(columnName)).count() == df.count(),
      "na values have been filled"
    )

  }

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
        StructField("label", IntegerType, nullable = true),
        StructField("automl_internal_id", LongType, nullable = true)
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
        StructField("label", IntegerType, nullable = true),
        StructField("automl_internal_id", LongType, nullable = true)
      )
    )

    val data =
      DiscreteTestDataGenerator.generateNAFillData(NA_FILL_ROW_COUNT, NA_RATE)

    val sanitizer = setupNAFillTest(numFillStat, catFillStat, mode)

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
    // Make sure that Boolean types have been filled.
    fillMap.booleanColumns.keys.toArray
      .map(x => naFilledDF.select(x).na.drop().count())
      .foreach { x =>
        assert(x == NA_FILL_ROW_COUNT, "for boolean fill characteristics")
      }

    // Ensure the model type is correct
    assert(modelType == EXPECTED_MODEL_TYPE, "for model type detection")

    val fullColumnCheck = fillMap.categoricalColumns.keys.toSeq ++ fillMap.numericColumns.keys.toSeq ++ fillMap.booleanColumns.keys.toSeq
    fullColumnCheck.foreach { x =>
      checkForNulls(naFilledDF, x)
    }

  }

  it should "correctly fill na's in auto mode, numeric mean, categorical max" in {

    val NUM_FILL_STAT = "mean"
    val CAT_FILL_STAT = "max"

    val expectedContinuousFillMap = Map(
      "dblData" -> 101.0,
      "fltData" -> 102.0,
      "intData" -> 99.22222222222223,
      "ordinalIntData" -> 15.311111111111112
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
      "dblData" -> 199.0,
      "fltData" -> 200.0,
      "intData" -> 199.0,
      "ordinalIntData" -> 25.0
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
      "dblData" -> 99.0,
      "fltData" -> 100.0,
      "intData" -> 99.0,
      "ordinalIntData" -> 17.0
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

  it should "correctly fill na's in mapFill mode" in {

    val FILL_MODE = "mapFill"

    val (naFilledDF, fillMap, modelType) = setupMapFillModes(FILL_MODE)

    NUMERIC_FILL_MAP.keys.foreach { x =>
      assert(
        naFilledDF.filter(col(x) === NUMERIC_FILL_MAP(x)).count() > 0,
        "for numeric fill columns"
      )

    }

    STRING_FILL_MAP.keys.foreach { x =>
      assert(
        naFilledDF.filter(col(x) === STRING_FILL_MAP(x)).count() > 0,
        "for categorical fill columns"
      )
    }

    val fullColumnCheck = fillMap.categoricalColumns.keys.toSeq ++ fillMap.numericColumns.keys.toSeq ++ fillMap.booleanColumns.keys.toSeq

    fullColumnCheck.foreach { x =>
      checkForNulls(naFilledDF, x)
    }

  }

  it should "correctly fill na's in blanketFillAll mode" in {
    val FILL_MODE = "blanketFillAll"

    val (naFilledDF, fillMap, modelType) = setupMapFillModes(FILL_MODE)

    NUMERIC_FILL_MAP.keys.foreach { x =>
      assert(
        naFilledDF.filter(col(x) === NUM_FILL_VALUE).count() > 0,
        "for numeric map fill"
      )
    }

    STRING_FILL_MAP.keys.foreach { x =>
      assert(
        naFilledDF.filter(col(x) === CHAR_FILL_VALUE).count() > 0,
        "for categorical map fill"
      )
    }

    val fullColumnCheck = fillMap.categoricalColumns.keys.toSeq ++ fillMap.numericColumns.keys.toSeq ++ fillMap.booleanColumns.keys.toSeq
    fullColumnCheck.foreach { x =>
      checkForNulls(naFilledDF, x)
    }
  }

  it should "correctly fill na's in blanketFillCharOnly mode" in {
    val FILL_MODE = "blanketFillCharOnly"
    val expectedContinuousFillMap = Map(
      "dblData" -> 101.0,
      "fltData" -> 102.0,
      "intData" -> 99.22222222222223,
      "ordinalIntData" -> 15.311111111111112
    )

    val (naFilledDF, fillMap, modelType) = setupMapFillModes(FILL_MODE)

    assert(
      fillMap.numericColumns == expectedContinuousFillMap,
      "for numeric fill na values"
    )
    STRING_FILL_MAP.keys.foreach { x =>
      assert(
        naFilledDF.filter(col(x) === CHAR_FILL_VALUE).count() > 0,
        "for categorical map fill"
      )
    }

    val fullColumnCheck = fillMap.categoricalColumns.keys.toSeq ++ fillMap.numericColumns.keys.toSeq ++ fillMap.booleanColumns.keys.toSeq
    fullColumnCheck.foreach { x =>
      checkForNulls(naFilledDF, x)
    }

  }
  it should "correctly fill na's in blanketFillNumOnly" in {
    val FILL_MODE = "blanketFillNumOnly"
    val expectedCategoricalFillMap = Map("strData" -> "e")
    val (naFilledDF, fillMap, modelType) = setupMapFillModes(FILL_MODE)

    assert(
      fillMap.categoricalColumns == expectedCategoricalFillMap,
      "for categorical fill na values"
    )
    NUMERIC_FILL_MAP.keys.foreach { x =>
      assert(
        naFilledDF.filter(col(x) === NUM_FILL_VALUE).count() > 0,
        "for numeric map fill"
      )
    }

    val fullColumnCheck = fillMap.categoricalColumns.keys.toSeq ++ fillMap.numericColumns.keys.toSeq ++ fillMap.booleanColumns.keys.toSeq
    fullColumnCheck.foreach { x =>
      checkForNulls(naFilledDF, x)
    }

  }

}
