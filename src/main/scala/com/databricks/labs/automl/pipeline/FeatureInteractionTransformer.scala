package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

class FeatureInteractionTransformer(override val uid: String)
    extends AbstractTransformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeatureColumn
    with HasTransformCalculated {

  def this() = {
    this(Identifiable.randomUID("FeatureInteractionTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)

  }

  final val modelingType: Param[String] =
    new Param[String](this, "modelingType", "Either regressor or classifier")
  final val nominalFields: StringArrayParam = new StringArrayParam(
    this,
    "nominalFields",
    "Fields that have been StringIndexed"
  )
  final val continuousFields: StringArrayParam = new StringArrayParam(
    this,
    "continuousFields",
    "Fields that are numeric in raw data set"
  )
  final val retentionMode: Param[String] = new Param[String](
    this,
    "retentionMode",
    "One of: 'all', 'optimistic', or 'strict' for interacted field inclusion"
  )
  final val continuousDiscretizerBucketCount: IntParam = new IntParam(
    this,
    "continuousDiscretizerBucketCount",
    "Quantization value for continuous features for entropy calculations"
  )
  final val featureInteractionParallelism: IntParam = new IntParam(
    this,
    "featureInteractionParallelism",
    "Number of concurrent processes to utilize while calculating interaction improvements to " +
      "InformationGain or Variance reduction"
  )
  final val targetInteractionPercentage: DoubleParam = new DoubleParam(
    this,
    "targetInteractionPercentage",
    "Threshold for inclusion based on interacted column comparison to parents"
  )
  final val interactionColumns: StringArrayParam = new StringArrayParam(
    this,
    "interactionColumns",
    "Columns to pairwise interact"
  )
  final val stringIndexColumns: StringArrayParam = new StringArrayParam(
    this,
    "stringIndexColumns",
    "Columns that need to be StringIndexed after interaction"
  )

  def setModelingType(value: String): this.type = set(modelingType, value)
  def getModelingType: String = $(modelingType)

  def setNominalFields(value: Array[String]): this.type =
    set(nominalFields, value)
  def getNominalFields: Array[String] = $(nominalFields)

  def setContinuousFields(value: Array[String]): this.type =
    set(continuousFields, value)
  def getContinuousFields: Array[String] = $(continuousFields)

  def setRetentionMode(value: String): this.type = set(retentionMode, value)
  def getRetentionMode: String = $(retentionMode)

  def setContinuousDiscretizerBucketCount(value: Int): this.type =
    set(continuousDiscretizerBucketCount, value)
  def getContinuousDiscretizerBucketCount: Int =
    $(continuousDiscretizerBucketCount)

  def setFeatureInteractionParallelism(value: Int): this.type =
    set(featureInteractionParallelism, value)
  def getFeatureInteractionParallelism: Int = $(featureInteractionParallelism)

  def setTargetInteractionPercentage(value: Double): this.type =
    set(targetInteractionPercentage, value)
  def getTargetInteractionPercentage: Double = $(targetInteractionPercentage)

//  def setFinalVectorFields(value: Array[String]): this.type = set(finalVectorFields)

  def setInteractionColumns(value: Array[String]): this.type =
    set(interactionColumns, value)

  override def transformInternal(dataset: Dataset[_]): DataFrame = ???

  override def transformSchemaInternal(schema: StructType): StructType = ???

  override def copy(extra: ParamMap): Transformer = ???
}
