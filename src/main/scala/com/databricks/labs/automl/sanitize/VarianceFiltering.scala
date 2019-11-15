package com.databricks.labs.automl.sanitize

import com.databricks.labs.automl.pipeline.FeaturePipeline
import org.apache.log4j.Logger
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer

class VarianceFiltering(data: DataFrame) {

  private var _labelCol = "label"
  private var _featureCol = "features"
  private var _dateTimeConversionType = "split"
  private var _parallelism = 20

  private val logger: Logger = Logger.getLogger(this.getClass)

  private final val dfSchema = data.schema.fieldNames

  def setLabelCol(value: String): this.type = {
    require(
      dfSchema.contains(value),
      s"Label Column $value does not exist in Dataframe"
    )
    _labelCol = value
    this
  }

  def setFeatureCol(value: String): this.type = {
    _featureCol = value
    this
  }

  def setDateTimeConversionType(value: String): this.type = {
    _dateTimeConversionType = value
    this
  }

  def setParallelism(value: Int): this.type = {
    _parallelism = value
    this
  }

  def getLabelCol: String = _labelCol

  def getFeatureCol: String = _featureCol

  def getDateTimeConversionType: String = _dateTimeConversionType

  def getParallelism: Int = _parallelism

  private def regenerateSchema(fieldSchema: Array[String]): Array[String] = {
    fieldSchema.map { x =>
      x.split("_si$")(0)
    }
  }

  private def getBatches(items: List[String]): Array[List[String]] = {
    val batches = ArrayBuffer[List[String]]()
    val batchSize = items.length / _parallelism
    for (i <- 0 to items.length by batchSize) {
      batches.append(items.slice(i, i + batchSize))
    }
    batches.toArray
  }

  def filterZeroVariance(
    fieldsToIgnore: Array[String] = Array.empty[String]
  ): (DataFrame, Array[String]) = {

    val (featurizedData, fields, allFields) = new FeaturePipeline(data)
      .setLabelCol(_labelCol)
      .setFeatureCol(_featureCol)
      .setDateTimeConversionType(_dateTimeConversionType)
      .makeFeaturePipeline(fieldsToIgnore)

    val dfParts = featurizedData.rdd.partitions.length.toDouble
    val summaryParts = Math.max(32, Math.min(Math.ceil(dfParts / 20.0).toInt, 200))
    val stddevInformation = featurizedData
      .coalesce(summaryParts)
      .summary("stddev")
      .select(fields map col: _*)
      .collect()(0)
      .toSeq
      .toArray

    val stddevData = fields.zip(stddevInformation)

    val preserveColumns = new ArrayBuffer[String]
    val removedColumns = new ArrayBuffer[String]

    stddevData.foreach { x =>
      if (x._2.toString.toDouble != 0.0) {
        preserveColumns.append(x._1)
      } else {
        removedColumns.append(x._1)
      }
    }

    //    val removedColumnsString = removedColumns.toArray.mkString(", ")
    //    println(s"The following columns were removed due to zero variance: $removedColumnsString")
    //
    //    logger.log(Level.WARN, s"The following columns were removed due to zero variance: $removedColumnsString")
    val finalFields = preserveColumns.result ++ Array(_labelCol)

    (data.select(finalFields map col: _*), removedColumns.toArray)

  }

}
