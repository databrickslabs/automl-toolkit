package com.databricks.labs.automl

import com.databricks.labs.automl.utilities.{
  DataGeneratorUtilities,
  ModelDetectionSchema,
  NaFillTestSchema,
  OutlierTestSchema,
  VarianceTestSchema
}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object DiscreteTestDataGenerator extends DataGeneratorUtilities {

  final private val MLFLOWID_START = 1
  final private val MLFLOWID_STEP = 1
  final private val MLFLOW_ID_MODE = "ascending"

  def generateNAFillData(rows: Int, naRate: Int): DataFrame = {

    val spark = AutomationUnitTestsUtil.sparkSession

    val DOUBLES_START = 1.0
    val DOUBLES_STEP = 2.0
    val DOUBLES_MODE = "ascending"
    val FLOAT_START = 0.0f
    val FLOAT_STEP = 2.0f
    val FLOAT_MODE = "descending"
    val INT_START = 1
    val INT_STEP = 2
    val INT_MODE = "ascending"
    val ORD_START = 5
    val ORD_STEP = 4
    val ORD_MODE = "descending"
    val ORD_DISTINCT_COUNT = 6
    val STRING_DISTINCT_COUNT = 5
    val DATE_YEAR = 2019
    val DATE_MONTH = 7
    val DATE_DAY = 25
    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "random"
    val LABEL_DISTINCT_COUNT = 4

    import spark.implicits._

    val targetNaModulus = rows / naRate

    val doublesSpace = generateDoublesDataWithNulls(
      rows,
      DOUBLES_START,
      DOUBLES_STEP,
      DOUBLES_MODE,
      targetNaModulus,
      0
    )

    val floatSpace = generateFloatsDataWithNulls(
      rows,
      FLOAT_START,
      FLOAT_STEP,
      FLOAT_MODE,
      targetNaModulus,
      1
    )

    val intSpace = generateIntDataWithNulls(
      rows,
      INT_START,
      INT_STEP,
      INT_MODE,
      targetNaModulus,
      2
    )
    val ordinalIntSpace = generateRepeatingIntDataWithNulls(
      rows,
      ORD_START,
      ORD_STEP,
      ORD_MODE,
      ORD_DISTINCT_COUNT,
      targetNaModulus,
      3
    )
    val stringSpace =
      generateStringDataWithNulls(
        rows,
        STRING_DISTINCT_COUNT,
        targetNaModulus,
        4
      )
    val booleanSpace = generateBooleanDataWithNulls(rows, targetNaModulus, 5)
    val daysSpace = generateDatesWithNulls(
      rows,
      DATE_YEAR,
      DATE_MONTH,
      DATE_DAY,
      targetNaModulus,
      6
    )
    val labelData = generateRepeatingIntData(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      LABEL_DISTINCT_COUNT
    )
    val mlFlowIdData = generateRepeatingIntData(
      rows,
      MLFLOWID_START,
      MLFLOWID_STEP,
      MLFLOW_ID_MODE,
      rows
    )

    val seqData = for (i <- 0 until rows)
      yield
        NaFillTestSchema(
          doublesSpace(i),
          floatSpace(i),
          intSpace(i),
          ordinalIntSpace(i),
          stringSpace(i),
          booleanSpace(i),
          daysSpace(i),
          labelData(i),
          mlFlowIdData(i)
        )
    val dfConversion = seqData
      .toDF()
      .withColumn("dateData", to_date(col("dateData"), "yyyy-MM-dd"))

    reassignToNulls(dfConversion)

  }

  def generateModelDetectionData(rows: Int, uniqueLabels: Int): DataFrame = {

    val FEATURE_START = 2.0
    val FEATURE_STEP = 1.0
    val FEATURE_MODE = "ascending"
    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "random"

    val spark = AutomationUnitTestsUtil.sparkSession

    val featureData =
      generateDoublesData(rows, FEATURE_START, FEATURE_STEP, FEATURE_MODE)
    val labelData = generateRepeatingIntData(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      uniqueLabels
    ).map(_.toDouble)

    val mlFlowIdData = generateRepeatingIntData(
      rows,
      MLFLOWID_START,
      MLFLOWID_STEP,
      MLFLOW_ID_MODE,
      rows
    )

    val seqData = for (i <- 0 until rows)
      yield ModelDetectionSchema(featureData(i), labelData(i), mlFlowIdData(i))

    import spark.implicits._

    seqData.toDF()

  }

  def generateOutlierData(rows: Int, uniqueLabels: Int): DataFrame = {

    val EXPONENTIAL_START = 0.0
    val EXPONENTIAL_STEP = 2.0
    val EXPONENTIAL_MODE = "ascending"
    val EXPONENTIAL_POWER = 3
    val LINEAR_START = 0.0
    val LINEAR_STEP = 1.0
    val LINEAR_MODE = "descending"
    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "random"
    val LABEL_DISTINCT = 5
    val EXPONENTIAL_TAIL_START = 1.0
    val EXPONENTIAL_TAIL_STEP = 5.0
    val EXPONENTIAL_TAIL_MODE = "ascending"
    val EXPONENTIAL_TAIL_POWER = 2

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._
    val a = generateExponentialData(
      rows,
      EXPONENTIAL_START,
      EXPONENTIAL_STEP,
      EXPONENTIAL_MODE,
      EXPONENTIAL_POWER
    )
    val b = generateDoublesData(rows, LINEAR_START, LINEAR_STEP, LINEAR_MODE)
    val c = generateTailedExponentialData(
      rows,
      EXPONENTIAL_TAIL_START,
      EXPONENTIAL_TAIL_STEP,
      EXPONENTIAL_TAIL_MODE,
      EXPONENTIAL_TAIL_POWER
    )
    val label = generateRepeatingIntData(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      LABEL_DISTINCT
    )
    val mlFlowIdData = generateRepeatingIntData(
      rows,
      MLFLOWID_START,
      MLFLOWID_STEP,
      MLFLOW_ID_MODE,
      rows
    )

    val seqData = for (i <- 0 until rows)
      yield OutlierTestSchema(a(i), b(i), c(i), label(i), mlFlowIdData(i))

    seqData.toDF()

  }

  def generateVarianceFilteringData(rows: Int): DataFrame = {

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    val DOUBLE_SERIES_START = 1.0
    val DOUBLE_SERIES_STEP = 1.0
    val DOUBLE_SERIES_MODE = "ascending"
    val REPEATING_DOUBLE_VALUE = 42.42
    val REPEATING_INT_VALUE = 9
    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "random"
    val LABEL_DISTINCT_COUNT = 7

    val a = generateDoublesData(
      rows,
      DOUBLE_SERIES_START,
      DOUBLE_SERIES_STEP,
      DOUBLE_SERIES_MODE
    )
    val b = generateFibonacciData(rows)
    val c = generateStaticDoubleSeries(rows, REPEATING_DOUBLE_VALUE)
    val d = generateStaticIntSeries(rows, REPEATING_INT_VALUE)
    val label = generateRepeatingIntData(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      LABEL_DISTINCT_COUNT
    )
    val mlflowIdData = generateRepeatingIntData(
      rows,
      MLFLOWID_START,
      MLFLOWID_STEP,
      MLFLOW_ID_MODE,
      rows
    )

    val seqData = for (i <- 0 until rows)
      yield
        VarianceTestSchema(a(i), b(i), c(i), d(i), label(i), mlflowIdData(i))

    seqData.toDF()
  }

}
