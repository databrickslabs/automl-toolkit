package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.sanitize.DataSanitizer
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}

class NAFillTest extends AbstractUnitSpec {

  private final val LABEL_COL = "label"
  private final val FEATURE_COL = "features"
  private final val DISTINCT_THRESHOLD = 10
  private final val PARALLELISM = 4
  private final val FILTER_PRECISION = 0.01

  it should "correctly fill missing Numeric Values with the column mean" in {

    val data = DiscreteTestDataGenerator.generateNAFillData(100, 10)

    data.show(5)

    val sanitizer = new DataSanitizer(data)
      .setLabelCol(LABEL_COL)
      .setFeatureCol(FEATURE_COL)
      .setModelSelectionDistinctThreshold(DISTINCT_THRESHOLD)
      .setNumericFillStat("mean") //TODO: test all modes
      .setCharacterFillStat("max") // TODO: test all modes
      .setParallelism(PARALLELISM)
      .setCategoricalNAFillMap(Map.empty[String, String])
      .setCharacterNABlanketFillValue("")
      .setNumericNABlanketFillValue(Double.NaN)
      .setNumericNAFillMap(Map.empty[String, Double])
      .setNAFillMode("auto") //TODO: test -> "auto", "mapFill", "blanketFillAll", "blanketFillCharOnly", "blanketFillNumOnly"
      .setFilterPrecision(FILTER_PRECISION)
      .setFieldsToIgnoreInVector(
        Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
      )

    val output = sanitizer.extractTypes(
      data,
      LABEL_COL,
      Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    )

    println(s"vector fields: ${output._1.mkString(",")}")
    println(s"conversion fields: ${output._2.mkString(", ")}")
    println(s"date fields: ${output._3.mkString(", ")}")

//    val (naFilledDF, fillMap, modelType) = sanitizer.generateCleanData()
//
//    data.show(100)
//
//    naFilledDF.show(100)

  }

}
