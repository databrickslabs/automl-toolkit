package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, SchemaUtils}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsWritable,
  Identifiable
}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Transformer for creating interacted feature fields based on FeatureInteraction module
  * @param uid Stage Identifier
  * @since 0.6.2
  * @author Ben Wilson, Databricks
  */
class InteractionTransformer(override val uid: String)
    extends AbstractTransformer
    with DefaultParamsWritable
    with HasInteractionColumns {

  def this() = {
    this(Identifiable.randomUID("InteractionTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setDebugEnabled(false)
  }

  def setLeftCols(value: Array[String]): this.type = set(leftColumns, value)
  def setRightCols(value: Array[String]): this.type = set(rightColumns, value)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    var data = dataset
    transformSchemaInternal(dataset.schema)
    if (SchemaUtils.isNotEmpty(getInteractionColumns)) {
      getInteractionColumns.foreach { x =>
        data = data.withColumn(s"i_${x._1}_${x._2}", col(x._1) * col(x._2))
      }
    }
    data.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {

    if (SchemaUtils.isNotEmpty(getInteractionColumns)) {
      val newFields = getInteractionColumns.map(x => {
        StructField(s"i_${x._1}_${x._2}", DoubleType)
      })
      StructType(schema.fields ++ newFields)
    } else schema

  }

  override def copy(extra: ParamMap): InteractionTransformer =
    defaultCopy(extra)

}

object InteractionTransformer
    extends DefaultParamsReadable[InteractionTransformer] {
  override def load(path: String): InteractionTransformer = super.load(path)
}
