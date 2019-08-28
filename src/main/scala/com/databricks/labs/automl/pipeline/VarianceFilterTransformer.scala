package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.SchemaUtils
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable.ArrayBuffer

class VarianceFilterTransformer(override val uid: String)
  extends Transformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeatureColumn
    with HasTransformCalculated {

  def this() = this(Identifiable.randomUID("VarianceFilterTransformer"))

  final val preserveColumns: StringArrayParam = new StringArrayParam(this, "preserveColumns", "Columns Preserved")

  final val removedColumns: StringArrayParam = new StringArrayParam(this, "removedColumns", "Columns Removed")

  def setPreserveColumns(value: Array[String]): this.type = set(preserveColumns, value)

  def getPreserveColumns: Array[String] = $(preserveColumns)

  def setRemovedColumns(value: Array[String]): this.type = set(removedColumns, value)

  def getRemovedColumns: Array[String] = $(removedColumns)


  override def transform(dataset: Dataset[_]): DataFrame = {

    if(!getTransformCalculated) {
      // Get columns without label and feature column
      val fields = dataset.columns.filterNot(field => Array(getLabelColumn, getFeatureCol).contains(field) )

      val dfParts = dataset.rdd.partitions.length.toDouble
      val summaryParts = Math.min(Math.ceil(dfParts / 20.0).toInt, 200)
      val stddevInformation = dataset.coalesce(summaryParts).summary("stddev")
        .select(fields map col: _*).collect()(0).toSeq.toArray

      val stddevData = fields.zip(stddevInformation)

      val preserveColumns = new ArrayBuffer[String]
      val removedColumns = new ArrayBuffer[String]

      stddevData.foreach { x =>
        if (x._2.toString.toDouble != 0.0){
          preserveColumns.append(x._1)
        } else {
          removedColumns.append(x._1)
        }
      }

      setPreserveColumns(preserveColumns.toArray)
      setRemovedColumns(removedColumns.toArray)
      setTransformCalculated(true)

      val finalFields = preserveColumns.result ++ Array(getLabelColumn)

      transformSchema(dataset.schema)
      return dataset.select(finalFields map col:_*).toDF()

    } else {
        if(SchemaUtils.isNotEmpty(getPreserveColumns.toList)) {
          val selectFields = getPreserveColumns ++ Array(getLabelColumn)
          transformSchema(dataset.schema)
          return dataset.select(selectFields map col:_*).toDF()
        }
    }
    transformSchema(dataset.schema)
    return dataset.toDF()
  }

  override def transformSchema(schema: StructType): StructType = {
    if(SchemaUtils.isNotEmpty(getRemovedColumns.toList)) {
      return StructType(schema.fields.filterNot(field => getRemovedColumns.contains(field.name)))
    }
    schema
  }

  override def copy(extra: ParamMap): VarianceFilterTransformer = defaultCopy(extra)

}

object VarianceFilterTransformer extends DefaultParamsReadable[VarianceFilterTransformer] {

  override def load(path: String): VarianceFilterTransformer = super.load(path)

}
