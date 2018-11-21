package com.databricks.spark.automatedml

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer

class VarianceFiltering(data: DataFrame) {

  private var _labelCol = "label"

  private final val dfSchema = data.schema.fieldNames

  def setLabelCol(value: String): this.type = {
    require(dfSchema.contains(value), s"Label Column $value does not exist in Dataframe")
    _labelCol = value
    this
  }

  def getLabelCol: String = _labelCol

  private def regenerateSchema(fieldSchema: Array[String]): Array[String] = {
    fieldSchema.map { x => x.split("_si$")(0) }
  }

  def filterZeroVariance(): DataFrame = {

    val (featurizedData, fields) = new FeaturePipeline(data).makeFeaturePipeline()

    val stddevInformation = featurizedData.summary().filter(col("summary") === "stddev")
      .select(fields map col: _*).collect()(0).toSeq.toArray

    val stddevData = fields.zip(stddevInformation)

    val preserveColumns = new ArrayBuffer[String]

    stddevData.foreach { x =>
      if (x._2.toString.toDouble != 0.0) preserveColumns += x._1
    }

    preserveColumns += _labelCol

    val selectableColumns = regenerateSchema(preserveColumns.toArray)

    data.select(selectableColumns map col: _*)

  }
}
