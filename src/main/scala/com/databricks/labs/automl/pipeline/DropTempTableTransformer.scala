package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

/**
  * @author Jas Bali
  * A [[WithNoopsStage]] transformer stage that is helpful when a previous stage
  * registers a temp table and is no longer required for the rest of the pipeline.
  * Supposed to be used with [[RegisterTempTableTransformer]] and [[DatasetsUnionTransformer]]
  * @param uid
  */
class DropTempTableTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with WithNoopsStage {

  final val tempTableName = new Param[String](this, "tempTableName", "tempTableName")

  def setTempTableName(value: String): this.type = set(tempTableName, value)

  def getTempTableName: String = $(tempTableName)

  def this() = {
    this(Identifiable.randomUID("DropTempTableTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setDebugEnabled(false)
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