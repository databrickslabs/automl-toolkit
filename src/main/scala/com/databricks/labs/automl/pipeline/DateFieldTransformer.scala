package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.{
  AutoMlPipelineMlFlowUtils,
  DataValidation,
  SchemaUtils
}
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsWritable,
  Identifiable
}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer

/**
  * @author Jas Bali
  * A transformer stage that breaks down date and field columns into the following feature columns:
  * Date = day, month, year
  * time = day, month, year, hour, minutes, seconds
  */
class DateFieldTransformer(override val uid: String)
    extends AbstractTransformer
    with DefaultParamsWritable
    with DataValidation
    with HasLabelColumn {

  def this() = {
    this(Identifiable.randomUID("DateFieldTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setNewDateTimeFeatureColumns(Array.empty)
    setOldDateTimeFeatureColumns(Array.empty)
    setDebugEnabled(false)
  }

  final val mode: Param[String] = new Param[String](
    this,
    "mode",
    "date/time conversion mode. Possible values 'split' and 'unix'"
  )

  final val newDateTimeFeatureColumns: StringArrayParam = new StringArrayParam(
    this,
    "newDateTimeFeatureColumns",
    "New Columns that were added for converting date/time features "
  )

  final val oldDateTimeFeatureColumns: StringArrayParam = new StringArrayParam(
    this,
    "oldDateTimeFeatureColumns",
    "Old Columns before converting date/time features"
  )

  def setMode(value: String): this.type = set(mode, value)

  def getMode: String = $(mode)

  def setNewDateTimeFeatureColumns(value: Array[String]): this.type =
    set(newDateTimeFeatureColumns, value)

  def getNewDateTimeFeatureColumns: Array[String] = $(newDateTimeFeatureColumns)

  def setOldDateTimeFeatureColumns(value: Array[String]): this.type =
    set(oldDateTimeFeatureColumns, value)

  def getOldDateTimeFeatureColumns: Array[String] = $(oldDateTimeFeatureColumns)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    val columnTypes = SchemaUtils.extractTypes(
      dataset.select(
        dataset.columns
          .filterNot(item => getAutomlInternalId.equals(item)) map col: _*
      ),
      getLabelColumn
    )
    if (columnTypes != null &&
        (SchemaUtils.isNotEmpty(columnTypes.dateFields) || SchemaUtils
          .isNotEmpty(columnTypes.timeFields))) {
      val dfWithDateTimeTransformedFeatures = convertDateAndTime(
        dataset.toDF(),
        columnTypes.dateFields,
        columnTypes.timeFields,
        getMode
      )
      val newDateTimeFeatureColumns =
        dfWithDateTimeTransformedFeatures._2.toArray[String]
      val columnsConvertedFrom = new ArrayBuffer[String]()
      if (SchemaUtils.isNotEmpty(columnTypes.dateFields)) {
        columnsConvertedFrom ++= columnTypes.dateFields
      }
      if (SchemaUtils.isNotEmpty(columnTypes.timeFields)) {
        columnsConvertedFrom ++= columnTypes.timeFields
      }
      setParamsIfEmptyInternal(
        newDateTimeFeatureColumns,
        columnsConvertedFrom.toArray
      )
      return dfWithDateTimeTransformedFeatures._1.drop(columnsConvertedFrom: _*)
    }
    dataset.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    if (SchemaUtils.isNotEmpty(getOldDateTimeFeatureColumns)) {
      val allCols = schema.fields.map(field => field.name)
      val missingDateTimeCols =
        getOldDateTimeFeatureColumns.filterNot(name => allCols.contains(name))
      if (missingDateTimeCols.nonEmpty) {
        throw new RuntimeException(
          s"""Following columns are missing: ${missingDateTimeCols.mkString(
            ", "
          )}"""
        )
      }
    }
    if (SchemaUtils.isNotEmpty(getNewDateTimeFeatureColumns)) {
      val newFields: Array[StructField] = getNewDateTimeFeatureColumns.map(
        colName => StructField(colName, IntegerType)
      )
      return StructType(
        schema.fields
          .filterNot(field => getOldDateTimeFeatureColumns.contains(field.name))
          ++
            newFields
      )
    }
    schema
  }

  private def setParamsIfEmptyInternal(
    newDateTimeFeatureColumns: Array[String],
    oldDateTimeFeatureColumns: Array[String]
  ): Unit = {
    if (SchemaUtils.isEmpty(getNewDateTimeFeatureColumns)) {
      setNewDateTimeFeatureColumns(newDateTimeFeatureColumns)
    }
    if (SchemaUtils.isEmpty(getOldDateTimeFeatureColumns)) {
      setOldDateTimeFeatureColumns(oldDateTimeFeatureColumns)
    }
  }

  override def copy(extra: ParamMap): DateFieldTransformer = defaultCopy(extra)
}

object DateFieldTransformer
    extends DefaultParamsReadable[DateFieldTransformer] {
  override def load(path: String): DateFieldTransformer = super.load(path)
}
