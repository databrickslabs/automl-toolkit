package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, SchemaUtils}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsWritable,
  Identifiable
}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.col

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

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    var data = dataset
    if (SchemaUtils.isNotEmpty(getInteractionColumns)) {
      getInteractionColumns.foreach { x =>
        val suffix =
          if (x._1.endsWith("_si") && x._2.endsWith("_si")) "_si" else ""
        data.withColumn(s"i_${x._1}_${x._2}$suffix", col(x._1) * col(x._2))
      }
    }
    data.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {

    if (SchemaUtils.isNotEmpty(getInteractionColumns)) {
      val newFields = getInteractionColumns.map(x => {
        val suffix =
          if (x._1.endsWith("_si") && x._2.endsWith("_si")) "_si" else ""
        StructField(s"i_${x._1}_${x._2}$suffix", DoubleType)
      })

      return StructType(schema.fields ++ newFields)
    } else schema

  }

  override def copy(extra: ParamMap): InteractionTransformer =
    defaultCopy(extra)

}

object InteractionTransformer
    extends DefaultParamsReadable[InteractionTransformer] {
  override def load(path: String): InteractionTransformer = super.load(path)
}
