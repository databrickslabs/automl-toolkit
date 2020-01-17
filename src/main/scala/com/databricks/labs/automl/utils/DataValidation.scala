package com.databricks.labs.automl.utils

import org.apache.log4j.Logger
import org.apache.spark.ml.feature.{
  OneHotEncoderEstimator,
  StringIndexer,
  VectorAssembler
}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ListBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

trait DataValidation {

  def _allowableDateTimeConversions = List("unix", "split")
  def _allowableCategoricalFilterModes = List("silent", "warn")
  def _allowableCardinalilties = List("approx", "exact")

  @transient lazy private val logger: Logger = Logger.getLogger(this.getClass)

  def invalidateSelection(value: String, allowances: Seq[String]): String = {
    s"${allowances.foldLeft("")((a, b) => a + " " + b)}"
  }

  def oneHotEncodeStrings(
    stringIndexedFields: List[String]
  ): (OneHotEncoderEstimator, Array[String]) = {

    var encodedColumns = new ListBuffer[String]
    var oneHotEncoders = new ListBuffer[OneHotEncoderEstimator]

    stringIndexedFields.foreach { x =>
      encodedColumns += x.dropRight(3) + "_oh"
    }

    val oneHotEncodeObj = new OneHotEncoderEstimator()
      .setHandleInvalid("keep")
      .setInputCols(stringIndexedFields.toArray)
      .setOutputCols(encodedColumns.result.toArray)

    (oneHotEncodeObj, encodedColumns.result.toArray)

  }

  def indexStrings(
    categoricalFields: List[String]
  ): (Array[StringIndexer], Array[String]) = {

    var indexedColumns = new ListBuffer[String]
    var stringIndexers = new ListBuffer[StringIndexer]

    categoricalFields.map(x => {
      val stringIndexedColumnName = x + "_si"
      val stringIndexerObj = new StringIndexer()
        .setHandleInvalid("keep")
        .setInputCol(x)
        .setOutputCol(stringIndexedColumnName)
      indexedColumns += stringIndexedColumnName
      stringIndexers += stringIndexerObj
    })

    (stringIndexers.result.toArray, indexedColumns.result.toArray)

  }

  private def splitDateTimeParts(
    df: DataFrame,
    dateFields: List[String],
    timeFields: List[String]
  ): (DataFrame, List[String]) = {

    var resultFields = new ListBuffer[String]

    var data = df
    dateFields.map(x => {
      data = data
        .withColumn(x + "_year", year(col(x)))
        .withColumn(x + "_month", month(col(x)))
        .withColumn(x + "_day", dayofmonth(col(x)))
      resultFields ++= List(x + "_year", x + "_month", x + "_day")
    })
    timeFields.map(x => {
      data = data
        .withColumn(x + "_year", year(col(x)))
        .withColumn(x + "_month", month(col(x)))
        .withColumn(x + "_day", dayofmonth(col(x)))
        .withColumn(x + "_hour", hour(col(x)))
        .withColumn(x + "_minute", minute(col(x)))
        .withColumn(x + "_second", second(col(x)))
      resultFields ++= List(
        x + "_year",
        x + "_month",
        x + "_day",
        x + "_hour",
        x + "_minute",
        x + "_second"
      )
    })

    (data, resultFields.result)

  }

  private def convertToUnix(
    df: DataFrame,
    dateFields: List[String],
    timeFields: List[String]
  ): (DataFrame, List[String]) = {

    var resultFields = new ListBuffer[String]

    var data = df

    dateFields.map(x => {
      data = data.withColumn(x + "_unix", unix_timestamp(col(x)).cast("Double"))
      resultFields += x + "_unix"
    })

    timeFields.map(x => {
      data = data.withColumn(x + "_unix", unix_timestamp(col(x)).cast("Double"))
      resultFields += x + "_unix"
    })

    (data, resultFields.result)

  }

  def convertDateAndTime(df: DataFrame,
                         dateFields: List[String],
                         timeFields: List[String],
                         mode: String): (DataFrame, List[String]) = {

    val (data, fieldList) = mode match {
      case "split" => splitDateTimeParts(df, dateFields, timeFields)
      case "unix"  => convertToUnix(df, dateFields, timeFields)
    }

    (data, fieldList)

  }

  def generateAssembly(
    numericColumns: List[String],
    characterColumns: List[String],
    featureCol: String
  ): (Array[StringIndexer], Array[String], VectorAssembler) = {

    val assemblerColumns = new ListBuffer[String]
    numericColumns.map(x => assemblerColumns += x)

    val (indexers, indexedColumns) = indexStrings(characterColumns)
    indexedColumns.map(x => assemblerColumns += x)

    val assembledColumns = assemblerColumns.result.toArray

    val assembler = new VectorAssembler()
      .setInputCols(assembledColumns)
      .setOutputCol(featureCol)

    (indexers, assembledColumns, assembler)
  }

  def validateLabelAndFeatures(df: DataFrame,
                               labelCol: String,
                               featureCol: String): Unit = {
    val dfSchema = df.schema
    assert(
      dfSchema.fieldNames.contains(labelCol),
      s"Dataframe does not contain label column named: $labelCol"
    )
    assert(
      dfSchema.fieldNames.contains(featureCol),
      s"Dataframe does not contain features column named: $featureCol"
    )
  }

  def validateFieldPresence(df: DataFrame, column: String): Unit = {
    val dfSchema = df.schema
    assert(
      dfSchema.fieldNames.contains(column),
      s"Dataframe does not contain column named: '$column'"
    )
  }

  def validateInputDataframe(df: DataFrame): Unit = {
    require(df != null, "Input dataset cannot be null")
    require(df.count() > 0, "Input dataset cannot be empty")
  }

  def validateCardinality(df: DataFrame,
                          stringFields: List[String],
                          cardinalityLimit: Int = 500,
                          parallelism: Int = 20): ValidatedCategoricalFields = {

    var validStringFields = ListBuffer[String]()
    var invalidStringFields = ListBuffer[String]()

    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(parallelism))
    val collection = stringFields.par
    collection.tasksupport = taskSupport

    collection.foreach { x =>
      val uniqueValues = df.select(x).distinct().count()
      if (uniqueValues <= cardinalityLimit) {
        validStringFields += x
      } else {
        invalidStringFields += x
      }
    }

    ValidatedCategoricalFields(
      validStringFields.toList,
      invalidStringFields.toList
    )

  }
}

case class ValidatedCategoricalFields(validFields: List[String],
                                      invalidFields: List[String])
