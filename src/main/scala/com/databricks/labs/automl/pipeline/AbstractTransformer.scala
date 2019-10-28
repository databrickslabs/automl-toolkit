package com.databricks.labs.automl.pipeline

import org.apache.log4j.Logger
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * @author Jas Bali
  * Abstract transformer should be extended for all AutoML transformers
  * This can contain common validation, exceptions and log messages.
  * Internally extends Spark Pipeline transformer [[Transformer]]
  */

abstract class AbstractTransformer
    extends Transformer
    with HasAutoMlIdColumn
    with HasDebug
    with HasPipelineId {

  @transient lazy private val logger: Logger = Logger.getLogger(this.getClass)

  /**
    * Final overridden method that cannot be modified by AutoML transformers
    * @param schema
    * @return Transformed Schema [[StructType]]
    */
  final override def transformSchema(schema: StructType): StructType = {
    transformSchemaInternal(schema)
  }

  /**
    * Final overridden method that cannot be modified by AutoML transformers
    *
    * @param dataset
    * @return Transformed DataFrame [[DataFrame]]
    */
  final override def transform(dataset: Dataset[_]): DataFrame = {
    val startMillis = System.currentTimeMillis()
    val outputDf =  transformInternal(dataset)
    transformSchemaInternal(dataset.schema)
    logAutoMlInternalIdPresent(outputDf)
    logTransformation(dataset, outputDf, System.currentTimeMillis() - startMillis)
    outputDf
  }

  final private def logAutoMlInternalIdPresent(outputDf: Dataset[_]): Unit = {
    val idAbsentMessage = s"Missing $getAutomlInternalId in the input columns"
    val isIdColumnNeeded = outputDf.schema.fieldNames.contains(getAutomlInternalId) || this.isInstanceOf[AutoMlOutputDatasetTransformer]
    if(!isIdColumnNeeded) {
      logger.fatal(s"idAbsentMessage in ${this.getClass}")
    }
    assert(isIdColumnNeeded, idAbsentMessage)
  }

  /**
    * Abstract Method to be implemented by all AutoML transformers
    * @param dataset
    * @return transformed output [[DataFrame]]
    */
  def transformInternal(dataset: Dataset[_]): DataFrame

  /**
    * Abstract Method to be implemented by all AutoML transformers
    * @param schema
    * @return schema of new output [[DataFrame]] [[StructType]]
    */
  def transformSchemaInternal(schema: StructType): StructType
}
