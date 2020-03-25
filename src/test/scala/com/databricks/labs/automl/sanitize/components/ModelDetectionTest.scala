package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.sanitize.DataSanitizer
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}

class ModelDetectionTest extends AbstractUnitSpec {

  private final val CLASSIFICATION_DISTINCT_COUNT = 3
  private final val REGRESSION_DISTINCT_COUNT = 11
  private final val ROW_COUNT = 500
  private final val LABEL_COL = "label"
  private final val FEATURE_COL = "features"
  private final val DISTINCT_THRESHOLD = 10
  private final val PARALLELISM = 4
  private final val FILTER_PRECISION = 0.01
  private final val STRING_FILL_MAP = Map.empty[String, String]
  private final val NUMERIC_FILL_MAP = Map.empty[String, AnyVal]
  private final val CHAR_FILL_VALUE = "hodor"
  private final val NUM_FILL_VALUE = 42.0
  private final val NUM_FILL_STAT = "mean"
  private final val CHAR_FILL_STAT = "max"
  private final val FILL_MODE = "auto"
  private final val REGRESSION_NAME = "regressor"
  private final val CLASSIFICATION_NAME = "classifier"

  it should "correctly identify a classification problem in auto mode" in {

    val classificationData =
      DiscreteTestDataGenerator.generateModelDetectionData(
        ROW_COUNT,
        CLASSIFICATION_DISTINCT_COUNT
      )

    val sanitizer = new DataSanitizer(classificationData)
      .setLabelCol(LABEL_COL)
      .setFeatureCol(FEATURE_COL)
      .setNumericFillStat(NUM_FILL_STAT)
      .setCharacterFillStat(CHAR_FILL_STAT)
      .setParallelism(PARALLELISM)
      .setCategoricalNAFillMap(STRING_FILL_MAP)
      .setCharacterNABlanketFillValue(CHAR_FILL_VALUE)
      .setNumericNABlanketFillValue(NUM_FILL_VALUE)
      .setNumericNAFillMap(NUMERIC_FILL_MAP)
      .setNAFillMode(FILL_MODE)
      .setFilterPrecision(FILTER_PRECISION)
      .setFieldsToIgnoreInVector(
        Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
      )

    val (data, fillMap, modelDecision) = sanitizer.generateCleanData()

    assert(
      modelDecision == CLASSIFICATION_NAME,
      "detect classification setting correctly"
    )

  }

  it should "correctly identify a regression problem in auto mode" in {

    val classificationData =
      DiscreteTestDataGenerator.generateModelDetectionData(
        ROW_COUNT,
        REGRESSION_DISTINCT_COUNT
      )

    val sanitizer = new DataSanitizer(classificationData)
      .setLabelCol(LABEL_COL)
      .setFeatureCol(FEATURE_COL)
      .setNumericFillStat(NUM_FILL_STAT)
      .setCharacterFillStat(CHAR_FILL_STAT)
      .setParallelism(PARALLELISM)
      .setCategoricalNAFillMap(STRING_FILL_MAP)
      .setCharacterNABlanketFillValue(CHAR_FILL_VALUE)
      .setNumericNABlanketFillValue(NUM_FILL_VALUE)
      .setNumericNAFillMap(NUMERIC_FILL_MAP)
      .setNAFillMode(FILL_MODE)
      .setFilterPrecision(FILTER_PRECISION)
      .setFieldsToIgnoreInVector(
        Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
      )

    val (data, fillMap, modelDecision) = sanitizer.generateCleanData()

    assert(
      modelDecision == REGRESSION_NAME,
      "detect regression setting correctly"
    )

  }

  it should "correctly identify a classification problem with overrides" in {

    val classificationData =
      DiscreteTestDataGenerator.generateModelDetectionData(
        ROW_COUNT,
        CLASSIFICATION_DISTINCT_COUNT
      )

    val sanitizer = new DataSanitizer(classificationData)
      .setLabelCol(LABEL_COL)
      .setFeatureCol(FEATURE_COL)
      .setModelSelectionDistinctThreshold(DISTINCT_THRESHOLD)
      .setNumericFillStat(NUM_FILL_STAT)
      .setCharacterFillStat(CHAR_FILL_STAT)
      .setParallelism(PARALLELISM)
      .setCategoricalNAFillMap(STRING_FILL_MAP)
      .setCharacterNABlanketFillValue(CHAR_FILL_VALUE)
      .setNumericNABlanketFillValue(NUM_FILL_VALUE)
      .setNumericNAFillMap(NUMERIC_FILL_MAP)
      .setNAFillMode(FILL_MODE)
      .setFilterPrecision(FILTER_PRECISION)
      .setFieldsToIgnoreInVector(
        Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
      )

    val (data, fillMap, modelDecision) = sanitizer.generateCleanData()

    assert(
      modelDecision == CLASSIFICATION_NAME,
      "detect classification setting correctly"
    )

  }

  it should "correctly identify a regression problem with overrides" in {

    val classificationData =
      DiscreteTestDataGenerator.generateModelDetectionData(
        ROW_COUNT,
        REGRESSION_DISTINCT_COUNT
      )

    val sanitizer = new DataSanitizer(classificationData)
      .setLabelCol(LABEL_COL)
      .setFeatureCol(FEATURE_COL)
      .setModelSelectionDistinctThreshold(DISTINCT_THRESHOLD)
      .setNumericFillStat(NUM_FILL_STAT)
      .setCharacterFillStat(CHAR_FILL_STAT)
      .setParallelism(PARALLELISM)
      .setCategoricalNAFillMap(STRING_FILL_MAP)
      .setCharacterNABlanketFillValue(CHAR_FILL_VALUE)
      .setNumericNABlanketFillValue(NUM_FILL_VALUE)
      .setNumericNAFillMap(NUMERIC_FILL_MAP)
      .setNAFillMode(FILL_MODE)
      .setFilterPrecision(FILTER_PRECISION)
      .setFieldsToIgnoreInVector(
        Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
      )

    val (data, fillMap, modelDecision) = sanitizer.generateCleanData()

    assert(
      modelDecision == REGRESSION_NAME,
      "detect regression setting correctly"
    )

  }

}
