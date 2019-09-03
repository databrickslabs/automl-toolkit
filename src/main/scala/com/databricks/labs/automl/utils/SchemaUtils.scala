package com.databricks.labs.automl.utils

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DataType, StructType}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

object SchemaUtils {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private def extractSchema(schema: StructType): List[(DataType, String)] = {

    var preParsedFields = new ListBuffer[(DataType, String)]

    schema.map(x => preParsedFields += ((x.dataType, x.name)))

    preParsedFields.result
  }

  def extractTypes(
                    data: DataFrame,
                    labelColumn: String
                  ): (List[String], List[String], List[String], List[String]) = {

    val fieldExtraction = extractSchema(data.schema)

    //DEBUG
    //println(s"EXTRACT TYPES field listing: ${fieldExtraction.map(x => x._2).mkString(", ")}")
    logger.log(
      Level.DEBUG,
      s"EXTRACT TYPES field listing: ${fieldExtraction.map(x => x._2).mkString(", ")}"
    )

    var conversionFields = new ListBuffer[String]
    var dateFields = new ListBuffer[String]
    var timeFields = new ListBuffer[String]
    var vectorizableFields = new ListBuffer[String]

    fieldExtraction.map(
      x =>
        x._1.typeName match {
          case "string"                    => conversionFields += x._2
          case "integer"                   => vectorizableFields += x._2
          case "double"                    => vectorizableFields += x._2
          case "float"                     => vectorizableFields += x._2
          case "long"                      => vectorizableFields += x._2
          case "byte"                      => conversionFields += x._2
          case "boolean"                   => vectorizableFields += x._2
          case "binary"                    => vectorizableFields += x._2
          case "date"                      => dateFields += x._2
          case "timestamp"                 => timeFields += x._2
          case z if z.take(7) == "decimal" => vectorizableFields += x._2
          case _ =>
            throw new UnsupportedOperationException(
              s"Field '${x._2}' is of type ${x._1} which is not supported."
            )
        }
    )

    vectorizableFields -= labelColumn

    (
      vectorizableFields.result,
      conversionFields.result,
      dateFields.result,
      timeFields.result
    )
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

  def isNotEmpty[A](list: Array[A]): Boolean = {
    list != null && list.nonEmpty
  }

  def isNotEmpty[A](list: List[A]): Boolean = {
    list != null && list.nonEmpty
  }


  def isEmpty[A](list: Array[A]): Boolean = {
    list == null || list.isEmpty
  }

  def generateStringIndexedColumn(columnName: String): String = {
    columnName + "_si"
  }

  def generateMapFromKeysValues[T](keys: Array[String],
                                values: Array[T]): Map[String, T] = {
    assert(keys.length == values.length, "Keys and Values lists cannot be different in size")
    var map = mutable.Map[String, T] ()
    for((key, i) <- keys.view.zipWithIndex) {
      map += (key -> values(i))
    }
    map.toMap
  }


}
