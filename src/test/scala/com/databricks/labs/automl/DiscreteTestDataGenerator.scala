package com.databricks.labs.automl

import com.databricks.labs.automl.utilities.{
  DataGeneratorUtilities,
  ModelDetectionSchema,
  NaFillTestSchema,
  OutlierTestSchema
}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object DiscreteTestDataGenerator extends DataGeneratorUtilities {

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
    val mlFlowIdData = generateRepeatingIntData(rows, 1, 1, "ascending", rows)

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

    val mlFlowIdData = generateRepeatingIntData(rows, 1, 1, "ascending", rows)

    val seqData = for (i <- 0 until rows)
      yield ModelDetectionSchema(featureData(i), labelData(i), mlFlowIdData(i))

    import spark.implicits._

    seqData.toDF()

  }

//  def generateOutlierData(rows: Int): DataFrame = {
//
//    val EXPONENTIAL_START =
//
//    val spark = AutomationUnitTestsUtil.sparkSession
//
//
//
//    import spark.implicits._
//    val a = generateExponentialData(rows)
//
//  }

  def generateOutlierData: DataFrame = {
    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    // TODO: replace this with the standard data set methodology above!!!
    Seq(
      OutlierTestSchema(0.0, 9.0, 0.99, 2, 1L),
      OutlierTestSchema(1.0, 8.0, 10.99, 2, 1L),
      OutlierTestSchema(2.0, 7.0, 0.99, 2, 1L),
      OutlierTestSchema(3.0, 6.0, 10.99, 2, 1L),
      OutlierTestSchema(4.0, 5.0, 0.99, 3, 1L),
      OutlierTestSchema(5.0, 4.0, 10.99, 3, 1L),
      OutlierTestSchema(6.0, 3.0, 10.99, 3, 1L),
      OutlierTestSchema(10.0, 2.0, 20.99, 3, 1L),
      OutlierTestSchema(20.0, 1.0, 20.99, 4, 1L),
      OutlierTestSchema(30.0, 2.0, 20.99, 5, 1L),
      OutlierTestSchema(40.0, 3.0, 20.99, 4, 1L),
      OutlierTestSchema(50.0, 4.0, 40.99, 4, 1L),
      OutlierTestSchema(60.0, 5.0, 40.99, 5, 1L),
      OutlierTestSchema(100.0, 6.0, 30.99, 1, 2L),
      OutlierTestSchema(200.0, 7.0, 30.99, 1, 3L),
      OutlierTestSchema(300.0, 8.0, 20.99, 1, 4L),
      OutlierTestSchema(1000.0, 9.0, 10.99, 3, 5L),
      OutlierTestSchema(10000.0, 10.0, 10.99, 4, 6L),
      OutlierTestSchema(100000.0, 20.0, 10.99, 3, 7L),
      OutlierTestSchema(1000000.0, 25.0, 1000.99, 10000, 8L),
      OutlierTestSchema(5000000.0, 50.0, 1.0, 17, 10L)
    ).toDF()
  }

}
