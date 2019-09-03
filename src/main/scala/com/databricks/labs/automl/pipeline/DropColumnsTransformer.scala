package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.{AutoMlPipelineUtils, SchemaUtils}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

class DropColumnsTransformer (override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with HasInputCols {

  def this() = {
    this(Identifiable.randomUID("DropColumnsTransformer"))
    setAutomlInternalId(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
  }

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    if(SchemaUtils.isNotEmpty(getInputCols)) {
      return dataset.drop(getInputCols: _*)
    }
    dataset.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    if(SchemaUtils.isNotEmpty(getInputCols)) {
      return StructType(schema.fields.filterNot(field => getInputCols.contains(field.name)))
    }
    schema
  }


  override def copy(extra: ParamMap): DropColumnsTransformer = defaultCopy(extra)
}

object DropColumnsTransformer extends DefaultParamsReadable[DropColumnsTransformer] {

  override def load(path: String): DropColumnsTransformer = super.load(path)

}
