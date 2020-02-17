package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._

/**
  * @author Jas Bali
  * This transformer rounds up input columns of type Double to Whole Double.
  */
class RoundUpDoubleTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with HasTransformCalculated
    with HasInputCols {

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  def this() = {
    this(Identifiable.randomUID("RoundUpDoubleTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setTransformCalculated(false)
    setDebugEnabled(false)
  }

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    transformSchemaInternal(dataset.schema)
    var tmpDf = dataset
    getInputCols.foreach(item =>
      tmpDf = tmpDf.withColumn(item, round(col(item)))
    )
    tmpDf.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): RoundUpDoubleTransformer = defaultCopy(extra)
}

object RoundUpDoubleTransformer extends DefaultParamsReadable[RoundUpDoubleTransformer] {
  override def load(path: String): RoundUpDoubleTransformer = super.load(path)
}
