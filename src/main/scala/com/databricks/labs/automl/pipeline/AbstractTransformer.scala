package com.databricks.labs.automl.pipeline

import org.apache.log4j.Logger
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

abstract class AbstractTransformer
    extends Transformer
    with HasAutoMlIdColumn{

  private val logger: Logger = Logger.getLogger(this.getClass)

  override def transformSchema(schema: StructType): StructType = {
    transformSchemaInternal(schema)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputDf =  transformInternal(dataset)
    transformSchemaInternal(dataset.schema)
    assert(outputDf.schema.fieldNames.contains(getAutomlInternalId), s"Missing $getAutomlInternalId in the input columns")
    outputDf
  }

  def transformInternal(dataset: Dataset[_]): DataFrame

  def transformSchemaInternal(schema: StructType): StructType

}
