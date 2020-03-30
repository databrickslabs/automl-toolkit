package com.databricks.labs.automl.utils

import com.databricks.labs.automl.pipeline.PipelineEnums
import com.databricks.labs.automl.utils.structures.{
  FieldDefinitions,
  FieldTypes
}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

object SchemaUtils {

  private val logger: Logger = Logger.getLogger(this.getClass)

//  private def extractSchema(schema: StructType): List[(DataType, String)] = {
//
//    var preParsedFields = new ListBuffer[(DataType, String)]
//
//    schema.map(x => preParsedFields += ((x.dataType, x.name)))
//
//    preParsedFields.result
//  }

  /**
    * Method for extracting the data type and field name from the StructType of a DataFrame schema
    * @param schema Schema of the DataFrame
    * @return Array[FieldDefinitions] with the payload of (dataType: DataType, fieldName: String)
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def extractSchema(schema: StructType): Array[FieldDefinitions] = {
    schema.map(x => FieldDefinitions(x.dataType, x.name)).toArray
  }

  /**
    * Standardized Type Extraction and assignment to different collections for handling of the various primitive types
    * @param data DataFrame that is in need of analysis
    * @param labelColumn Label Column of the DataFrame
    * @return FieldTypes which contains the Lists of the column names based on their collective handling types
    * @since 0.1.0
    * @author Ben Wilson, DataBricks
    * @throws UnsupportedOperationException if the data type seen in the DataFrame is not currently supported.
    */
  @throws(classOf[UnsupportedOperationException])
  def extractTypes(
    data: DataFrame,
    labelColumn: String,
    fieldsToIgnore: Array[String] = Array.empty[String]
  ): FieldTypes = {

    val fieldExtraction = extractSchema(data.schema)
      .filterNot(x => fieldsToIgnore.contains(x.fieldName))

    logger.log(
      Level.DEBUG,
      s"EXTRACT TYPES field listing: ${fieldExtraction.map(x => x.fieldName).mkString(", ")}"
    )

    var categoricalFields = new ListBuffer[String]
    var dateFields = new ListBuffer[String]
    var timeFields = new ListBuffer[String]
    var numericFields = new ListBuffer[String]
    var booleanFields = new ListBuffer[String]

    fieldExtraction.map(
      x =>
        x.dataType.typeName match {
          case "string"                    => categoricalFields += x.fieldName
          case "integer"                   => numericFields += x.fieldName
          case "double"                    => numericFields += x.fieldName
          case "float"                     => numericFields += x.fieldName
          case "long"                      => numericFields += x.fieldName
          case "byte"                      => categoricalFields += x.fieldName
          case "boolean"                   => booleanFields += x.fieldName
          case "binary"                    => booleanFields += x.fieldName
          case "date"                      => dateFields += x.fieldName
          case "timestamp"                 => timeFields += x.fieldName
          case z if z.take(7) == "decimal" => numericFields += x.fieldName
          case _ =>
            throw new UnsupportedOperationException(
              s"Field '${x.fieldName}' is of type ${x.dataType} (${x.dataType.typeName}) which is not supported."
            )
      }
    )

    numericFields -= labelColumn

    FieldTypes(
      numericFields.result,
      categoricalFields.result,
      dateFields.result,
      timeFields.result,
      booleanFields.result
    )
  }

  def isLabelRefactorNeeded(schema: StructType, labelCol: String): Boolean = {
    val labelDataType = schema.fields.find(_.name.equals(labelCol)).get.dataType
    labelDataType.typeName match {
      case "string"                    => true
      case "integer"                   => false
      case "double"                    => false
      case "float"                     => false
      case "long"                      => false
      case "byte"                      => true
      case "boolean"                   => false
      case "binary"                    => false
      case "date"                      => true
      case "timestamp"                 => true
      case z if z.take(7) == "decimal" => true
      case _ =>
        throw new UnsupportedOperationException(
          s"Field '$labelCol' is of type $labelDataType, which is not supported."
        )
    }
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
    columnName + PipelineEnums.SI_SUFFIX.value
  }

  def generateOneHotEncodedColumn(columnName: String): String = {
    val oheSuffix = PipelineEnums.OHE_SUFFIX.value
    if (columnName.endsWith(PipelineEnums.SI_SUFFIX.value)) {
      columnName.dropRight(3) + oheSuffix
    } else {
      columnName + oheSuffix
    }
  }

  def generateMapFromKeysValues[T](keys: Array[String],
                                   values: Array[T]): Map[String, T] = {
    assert(
      keys.length == values.length,
      "Keys and Values lists cannot be different in size"
    )
    var map = mutable.Map[String, T]()
    for ((key, i) <- keys.view.zipWithIndex) {
      map += (key -> values(i))
    }
    map.toMap
  }
}
