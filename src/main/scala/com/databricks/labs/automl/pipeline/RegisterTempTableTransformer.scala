package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

/**
  * @author Jas Bali
  * A [[WithNoopsStage]] transformer stage that is helpful when a following pipeline stage needs access to more
  * than a single dataset. This is to bypass pipeline semantics of only being able to pass a single dataset.
  * Useful for [[SyntheticFeatureGenTransformer]] where original rows need to be appended with synthetic rows from
  * [[com.databricks.labs.automl.feature.KSampling]]
  * @param uid
  */
class RegisterTempTableTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable {
  final val tempTableName = new Param[String](this, "tempTableName", "tempTableName")
  final val statement = new Param[String](this, "statement", "statement")

  def setTempTableName(value: String): this.type = set(tempTableName, value)

  def getTempTableName: String = $(tempTableName)

  def setStatement(value: String): this.type = set(statement, value)

  def getStatement: String = $(statement)

  def this() = {
    this(Identifiable.randomUID("RegisterTempTableTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setDebugEnabled(false)
  }

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    val tmpTableName = Identifiable.randomUID("InternalRegisterTempTableTransformer_")
    dataset.createOrReplaceTempView(tmpTableName)
    dataset
      .sqlContext
      .sql(getStatement.replaceAll("__THIS__", tmpTableName))
      .createOrReplaceTempView(getTempTableName)
    dataset.sqlContext.dropTempTable(tmpTableName)
    dataset.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): RegisterTempTableTransformer = defaultCopy(extra)

}

object RegisterTempTableTransformer extends DefaultParamsReadable[RegisterTempTableTransformer] {
  override def load(path: String): RegisterTempTableTransformer = super.load(path)
}
