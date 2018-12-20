package com.databricks.spark.automatedml.utils

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}

import scala.collection.mutable.ListBuffer

trait DataValidation {

  def _allowableDateTimeConversions = List("unix", "split")
  private val logger: Logger = Logger.getLogger(this.getClass)

  def invalidateSelection(value: String, allowances: Seq[String]): String = {
    s"${allowances.foldLeft("")((a, b) => a + " " + b)}"
  }

  def extractSchema(schema: StructType): List[(DataType, String)] = {

    var preParsedFields = new ListBuffer[(DataType, String)]

    schema.map(x => preParsedFields += ((x.dataType, x.name)))

    preParsedFields.result
  }

  def extractTypes(data: DataFrame, labelColumn: String, ignoreFields: Array[String]):
  (List[String], List[String], List[String], List[String]) = {

    val fieldExtraction = extractSchema(data.schema).filterNot(x => ignoreFields.contains(x._2))

    //DEBUG
    //println(s"EXTRACT TYPES field listing: ${fieldExtraction.map(x => x._2).mkString(", ")}")
    logger.log(Level.DEBUG, s"EXTRACT TYPES field listing: ${fieldExtraction.map(x => x._2).mkString(", ")}")

    var conversionFields = new ListBuffer[String]
    var dateFields = new ListBuffer[String]
    var timeFields = new ListBuffer[String]
    var vectorizableFields = new ListBuffer[String]

    fieldExtraction.map(x =>
      x._1.typeName match {
        case "string" => conversionFields += x._2
        case "integer" => vectorizableFields += x._2
        case "double" => vectorizableFields += x._2
        case "float" => vectorizableFields += x._2
        case "long" => vectorizableFields += x._2
        case "byte" => conversionFields += x._2
        case "boolean" => vectorizableFields += x._2
        case "binary" => vectorizableFields += x._2
        case "date" => dateFields += x._2
        case "timestamp" => timeFields += x._2
        case _ => throw new UnsupportedOperationException(
          s"Field '${x._2}' is of type ${x._1} which is not supported.")
      }
    )

    assert(vectorizableFields.contains(labelColumn),
      s"The provided Dataframe MUST contain a labeled column with the name '$labelColumn'")
    vectorizableFields -= labelColumn

    (vectorizableFields.result, conversionFields.result, dateFields.result, timeFields.result)

  }

  def indexStrings(categoricalFields: List[String]): (Array[StringIndexer], Array[String]) = {

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

  private def splitDateTimeParts(df: DataFrame, dateFields: List[String], timeFields: List[String]):
  (DataFrame, List[String]) = {

    var resultFields = new ListBuffer[String]

    var data = df
    dateFields.map(x => {
      data = data.withColumn(x + "_year", year(col(x)))
        .withColumn(x + "_month", month(col(x)))
        .withColumn(x + "_day", dayofmonth(col(x)))
      resultFields ++= List(x + "_year", x + "_month", x + "_day")
    })
    timeFields.map(x => {
      data = data.withColumn(x + "_year", year(col(x)))
        .withColumn(x + "_month", month(col(x)))
        .withColumn(x + "_day", dayofmonth(col(x)))
        .withColumn(x + "_hour", hour(col(x)))
        .withColumn(x + "_minute", minute(col(x)))
        .withColumn(x + "_second", second(col(x)))
      resultFields ++= List(x + "_year", x + "_month", x + "_day", x + "_hour", x + "_minute", x + "_second")
    })

    (data, resultFields.result)

  }

  private def convertToUnix(df: DataFrame, dateFields: List[String], timeFields: List[String]):
    (DataFrame, List[String]) = {

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

  def convertDateAndTime(df: DataFrame, dateFields: List[String], timeFields: List[String], mode: String):
  (DataFrame, List[String]) = {

    val (data, fieldList) = mode match {
      case "split" => splitDateTimeParts(df, dateFields, timeFields)
      case "unix" => convertToUnix(df, dateFields, timeFields)
    }

    (data, fieldList)

  }

  def generateAssembly(numericColumns: List[String], characterColumns: List[String], featureCol: String):
  (Array[StringIndexer], Array[String], VectorAssembler) = {

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

  def validateLabelAndFeatures(df: DataFrame, labelCol: String, featureCol: String): Unit = {
    val dfSchema = df.schema
    assert(dfSchema.fieldNames.contains(labelCol),
      s"Dataframe does not contain label column named: $labelCol")
    assert(dfSchema.fieldNames.contains(featureCol),
      s"Dataframe does not contain features column named: $featureCol")
  }

  def validateFieldPresence(df: DataFrame, column: String): Unit = {
    val dfSchema = df.schema
    assert(dfSchema.fieldNames.contains(column), s"Dataframe does not contain column named: '$column'")
  }

}
