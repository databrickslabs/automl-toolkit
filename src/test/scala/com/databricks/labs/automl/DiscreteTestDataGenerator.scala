package com.databricks.labs.automl

import com.databricks.labs.automl.utilities.DataGeneratorUtilities
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.joda.time.LocalDate

object DiscreteTestDataGenerator extends DataGeneratorUtilities {

  // TODO: create a constructor pattern here with a configuration to setup the needed by-column configuration for generating the universal data set for different tests

  //TODO: move the rest of these and improve them to be more generic for usages in other test modules.
  private def generateFloats(targetCount: Int,
                             targetModulus: Int,
                             offset: Int) = {
    (0.0f to targetCount by 1.0f).toArray.zipWithIndex
      .map {
        case (v, i) =>
          if ((i + offset) % targetModulus != 0.0) v else Float.MinValue
      }
  }

  private def generateInts(targetCount: Int,
                           targetModulus: Int,
                           offset: Int) = {
    (0 to targetCount by 1).toArray.zipWithIndex
      .map {
        case (v, i) =>
          if ((i + offset) % targetModulus != 0.0) v
          else Int.MinValue
      }
  }

  private def generateRepeatingInts(targetCount: Int, distinctValues: Int) = {
    val baseCollection = (0 to distinctValues).toArray
    Array
      .fill(targetCount / (baseCollection.length - 1))(baseCollection)
      .flatten
      .take(targetCount)
  }

  private def generateOrdinalInts(targetCount: Int,
                                  targetModulus: Int,
                                  distinctValues: Int,
                                  offset: Int) = {
    generateRepeatingInts(targetCount, distinctValues).zipWithIndex.map {
      case (v, i) =>
        if ((i + offset) % targetModulus != 0.0) v else Int.MinValue
    }
  }

  private def generateBooleans(targetCount: Int,
                               targetModulus: Int,
                               offset: Int) = {
    Array
      .fill(targetCount)(Array(true, false))
      .flatten
      .take(targetCount)
      .zipWithIndex
      .map {
        case (v, i) =>
          if ((i + offset) % targetModulus != 0.0) v else null
      }
      .map(_.asInstanceOf[Boolean])
  }

  private def generateDates(targetCount: Int,
                            targetModulus: Int,
                            offset: Int) = {
    val start = new LocalDate(2019, 7, 25)
    val dates = for (x <- 0 to targetCount) yield start.plusDays(x)
    dates.map(_.toString).toArray.zipWithIndex.map {
      case (v, i) => if ((i + offset) % targetModulus != 0.0) v else null
    }
  }

  def generateNAFillData(rows: Int,
                         naRate: Int,
                         distinctStringCount: Int,
                         distinctOrdinalCount: Int): DataFrame = {

    val spark = AutomationUnitTestsUtil.sparkSession

    //TODO: move these!!!!
    val DOUBLES_START = 1.0
    val DOUBLES_STEP = 1.0
    val DOUBLES_MODE = "ascending"

    import spark.implicits._

    val targetNaModulus = rows / naRate

    val doublesSpace = generateDoublesArrayWithNulls(
      rows,
      DOUBLES_START,
      DOUBLES_STEP,
      DOUBLES_MODE,
      targetNaModulus,
      0
    )
    val floatSpace = generateFloats(rows, targetNaModulus, 1)
    val intSpace = generateInts(rows, targetNaModulus, 2)
    val ordinalIntSpace = generateOrdinalInts(rows, targetNaModulus, 4, 5)
    val stringSpace = generateStringArrayWithNulls(
      rows,
      distinctStringCount,
      targetNaModulus,
      3
    )
    val booleanSpace = generateBooleans(rows, targetNaModulus, 4)
    val daysSpace = generateDates(rows, targetNaModulus, 2)
    val labelData = generateRepeatingInts(rows, 3)
    val mlFlowIdData = generateRepeatingInts(rows, 8)

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
    seqData.toSeq
      .toDF()
      .withColumn(
        "dblData",
        when(col("dblData") === Double.MinValue, null).otherwise(col("dblData"))
      )
      .withColumn(
        "fltData",
        when(col("fltData") === Float.MinValue, null).otherwise(col("fltData"))
      )
      .withColumn(
        "intData",
        when(col("intData") === Int.MinValue, null)
          .otherwise(col("intData"))
      )
      .withColumn(
        "ordinalIntData",
        when(col("ordinalIntData") === Int.MinValue, null)
          .otherwise(col("ordinalIntData"))
      )
      .withColumn("dateData", to_date(col("dateData"), "yyyy-MM-dd"))

  }

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
