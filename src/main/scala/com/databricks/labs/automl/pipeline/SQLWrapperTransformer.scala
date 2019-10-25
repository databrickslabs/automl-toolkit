package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

/**
  * @author Jas Bali
  * This transformer wraps [[SQLTransformer]] and is useful to add logging capability from [[AbstractTransformer]]
  */
class SQLWrapperTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable {

  final val statement = new Param[String](this, "statement", "statement")

  def setStatement(value: String): this.type = set(statement, value)

  def getStatement: String = $(statement)

  def this() = {
    this(Identifiable.randomUID("SQLWrapperTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setDebugEnabled(false)
  }

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    transformSchemaInternal(dataset.schema)
    new SQLTransformer().setStatement(getStatement).transform(dataset)
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
   new SQLTransformer().setStatement(getStatement).transformSchema(schema)
  }

  override def copy(extra: ParamMap): SQLWrapperTransformer = defaultCopy(extra)

}

object SQLWrapperTransformer extends DefaultParamsReadable[SQLWrapperTransformer] {
  override def load(path: String): SQLWrapperTransformer = super.load(path)
}
