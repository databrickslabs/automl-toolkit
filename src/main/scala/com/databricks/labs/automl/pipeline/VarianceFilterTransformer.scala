package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, SchemaUtils}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable.ArrayBuffer

/**
  * @author Jas Bali
  * Input: Vectorized feature columns
  * Output: variance filtered DataFrame [[DataFrame]]
  */
class VarianceFilterTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeatureColumn
    with HasTransformCalculated {

  final val preserveColumns = new StringArrayParam(this, "preserveColumns", "Columns Preserved")

  final val removedColumns = new StringArrayParam(this, "removedColumns", "Columns Removed")

  def setPreserveColumns(value: Array[String]): this.type = set(preserveColumns, value)

  def getPreserveColumns: Array[String] = $(preserveColumns)

  def setRemovedColumns(value: Array[String]): this.type = set(removedColumns, value)

  def getRemovedColumns: Array[String] = $(removedColumns)

  def this() = {
    this(Identifiable.randomUID("VarianceFilterTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setPreserveColumns(Array.empty)
    setRemovedColumns(Array.empty)
    setTransformCalculated(false)
    setDebugEnabled(false)
  }

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    // Get columns without label,  feature column and automl_internal_id columns
    val colsToIgnoreForVariance = if(dataset.columns.contains(getLabelColumn)) {
      Array(getLabelColumn, getFeatureCol, getAutomlInternalId)
    } else {
      Array(getFeatureCol, getAutomlInternalId)
    }

    if(!getTransformCalculated) {
      val fields = dataset.columns.filterNot(field => colsToIgnoreForVariance.contains(field))

      val dfParts = dataset.rdd.partitions.length.toDouble
      val summaryParts = Math.max(32, Math.min(Math.ceil(dfParts / 20.0).toInt, 200))
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

      val finalFields = getPreserveColumns ++ colsToIgnoreForVariance
      return dataset.select(finalFields map col:_*).toDF()
    } else {
        if(SchemaUtils.isNotEmpty(getPreserveColumns.toList)) {
          val selectFields = getPreserveColumns ++ colsToIgnoreForVariance
          return dataset.select(selectFields map col:_*).toDF()
        }
    }
    dataset.drop(getRemovedColumns:_*)
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    if(schema.fieldNames.contains(getLabelColumn)) {
      if (SchemaUtils.isNotEmpty(getRemovedColumns.toList)) {
        return StructType(schema.fields.filterNot(field => getRemovedColumns.contains(field.name)))
      }
    }
    schema
  }

  override def copy(extra: ParamMap): VarianceFilterTransformer = defaultCopy(extra)
}

object VarianceFilterTransformer extends DefaultParamsReadable[VarianceFilterTransformer] {
  override def load(path: String): VarianceFilterTransformer = super.load(path)
}
