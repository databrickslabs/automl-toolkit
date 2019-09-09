package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

class DropTempTableTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable {

  final val tempTableName = new Param[String](this, "tempTableName", "tempTableName")

  def setTempTableName(value: String): this.type = set(tempTableName, value)

  def getTempTableName: String = $(tempTableName)

  def this() = {
    this(Identifiable.randomUID("DropTempTableTransformer"))
    setAutomlInternalId(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
  }

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    dataset.sqlContext.dropTempTable(getTempTableName)
    dataset.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): DropTempTableTransformer = defaultCopy(extra)

}

object DropTempTableTransformer extends DefaultParamsReadable[DropTempTableTransformer] {
  override def load(path: String): DropTempTableTransformer = super.load(path)
}