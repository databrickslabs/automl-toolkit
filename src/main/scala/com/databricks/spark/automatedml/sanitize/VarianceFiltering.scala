package com.databricks.spark.automatedml.sanitize

import com.databricks.spark.automatedml.pipeline.FeaturePipeline
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.Column
class VarianceFiltering(data: DataFrame) {

  private var _labelCol = "label"
  private var _featureCol = "features"
  private var _dateTimeConversionType = "split"

  private final val dfSchema = data.schema.fieldNames

  def setLabelCol(value: String): this.type = {
    require(dfSchema.contains(value), s"Label Column $value does not exist in Dataframe")
    _labelCol = value
    this
  }

  def setFeatureCol(value: String) : this.type = {
    _featureCol = value
    this
  }

  def setDateTimeConversionType(value: String): this.type = {
    _dateTimeConversionType = value
    this
  }

  def getLabelCol: String = _labelCol

  def getFeatureCol: String = _featureCol

  def getDateTimeConversionType: String = _dateTimeConversionType

  private def regenerateSchema(fieldSchema: Array[String]): Array[String] = {
    fieldSchema.map { x => x.split("_si$")(0) }
  }

  def filterZeroVariance(fieldsToIgnore: Array[String]=Array.empty[String]): DataFrame = {

    val (featurizedData, fields, allFields) = new FeaturePipeline(data)
              .setLabelCol(_labelCol)
              .setFeatureCol(_featureCol)
              .setDateTimeConversionType(_dateTimeConversionType)
              .makeFeaturePipeline(fieldsToIgnore)

    val stddevInformation = featurizedData.summary("stddev")
      .select(fields map col: _*).collect()(0).toSeq.toArray

    val stddevData = fields.zip(stddevInformation)

    val preserveColumns = new ArrayBuffer[String]

    stddevData.foreach { x =>
      if (x._2.toString.toDouble != 0.0) preserveColumns += x._1
    }

    val finalFields = preserveColumns.result ++ Array(_labelCol) ++ fieldsToIgnore

    data.select(finalFields map col:_*)

  }
}
