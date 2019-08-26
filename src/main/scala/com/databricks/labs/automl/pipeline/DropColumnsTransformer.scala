package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.SchemaUtils
import org.apache.spark.annotation.Since
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

class DropColumnsTransformer (override val uid: String)
  extends Transformer
    with DefaultParamsWritable
    with HasInputCols {

  def this() = this(Identifiable.randomUID("DropColumnsTransformer"))

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    if(SchemaUtils.isNotEmpty(getInputCols.toList)) {
      dataset.drop(getInputCols: _*)
    }
    dataset.toDF()
  }

  override def transformSchema(schema: StructType): StructType = {
    if(SchemaUtils.isNotEmpty(getInputCols.toList)) {
      StructType(schema.fields.filterNot(field => getInputCols.contains(field.name)))
    }
    schema
  }


  override def copy(extra: ParamMap): DropColumnsTransformer = defaultCopy(extra)
}

object DropColumnsTransformer extends DefaultParamsReadable[DropColumnsTransformer] {

  override def load(path: String): DropColumnsTransformer = super.load(path)

}
