package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, SchemaUtils}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * @author Jas Bali
  * A transformer stage that can drop columns from an input Dataset.
  * Necessary when there are intermediate stages that require columns to be
  * removed from an input dataset because they aren't needed in the downstream
  * stages anymore, such as input columns for SI are not needed for OHE, input cols for
  * VA etc
  */
class DropColumnsTransformer (override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with HasInputCols {

  def this() = {
    this(Identifiable.randomUID("DropColumnsTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setDebugEnabled(false)
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
