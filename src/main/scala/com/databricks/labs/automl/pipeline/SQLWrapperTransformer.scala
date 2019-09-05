package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

class SQLWrapperTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with HasTransformCalculated {

  final val statement = new Param[String](this, "statement", "statement")

  def setStatement(value: String): this.type = set(statement, value)

  def getStatement: String = $(statement)

  def this() = {
    this(Identifiable.randomUID("SQLWrapperTransformer"))
    setAutomlInternalId(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
    setTransformCalculated(false)
  }

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    transformSchemaInternal(dataset.schema)
    if(!getTransformCalculated) {
      setTransformCalculated(true)
      return new SQLTransformer().setStatement(getStatement).transform(dataset)
    }
    dataset.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    if(!getTransformCalculated) {
      return new SQLTransformer().setStatement(getStatement).transformSchema(schema)
    }
    schema
  }

  override def copy(extra: ParamMap): SQLWrapperTransformer = defaultCopy(extra)

}

object SQLWrapperTransformer extends DefaultParamsReadable[SQLWrapperTransformer] {
  override def load(path: String): SQLWrapperTransformer = super.load(path)
}
