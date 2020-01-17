package com.databricks.labs.automl.feature.structures

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame

case class ColumnTypeData(name: String, dataType: String)
case class VarianceData(labelValue: Double, variance: Double)
case class EntropyData(labelValue: Double, entropy: Double)
case class InteractionPayloadExtract(left: String,
                                     leftDataType: String,
                                     right: String,
                                     rightDataType: String,
                                     outputName: String,
                                     score: Double)
case class InteractionPayload(left: String,
                              leftDataType: String,
                              right: String,
                              rightDataType: String,
                              outputName: String)
case class ColumnScoreData(score: Double, dataType: String)
case class InteractionResult(left: String,
                             right: String,
                             interaction: String,
                             score: Double)
case class FeatureInteractionCollection(
  data: DataFrame,
  interactionPayload: Array[InteractionPayloadExtract]
)
case class FeatureInteractionOutputPayload(
  data: DataFrame,
  fullFeatureVectorColumns: Array[String],
  interactionReport: Array[InteractionPayloadExtract]
)
case class NominalIndexCollection(name: String, indexCheck: Boolean)
case class NominalDataCollection(data: DataFrame,
                                 adjustedFields: Array[String],
                                 fieldsToRemove: Array[String],
                                 indexers: Array[StringIndexer])
case class PipelineInteractionOutput(
  pipeline: Pipeline,
  data: DataFrame,
  fullFeatureVectorColumns: Array[String],
  interactionReport: Array[InteractionPayloadExtract]
)
case class VectorAssemblyOutput(assembler: VectorAssembler, data: DataFrame)

object ModelingType extends Enumeration {
  val Regressor = ModelType("regressor")
  val Classifier = ModelType("classifier")
  protected case class ModelType(modelType: String) extends super.Val()
  implicit def convert(value: Value): ModelType = value.asInstanceOf[ModelType]
}

object FieldEncodingType extends Enumeration {
  val Nominal = FieldType("nominal")
  val Continuous = FieldType("continuous")
  protected case class FieldType(fieldType: String) extends super.Val()
  implicit def convert(value: Value): FieldType = value.asInstanceOf[FieldType]
}

object InteractionRetentionMode extends Enumeration {
  val Optimistic = RetentionMode("optimistic")
  val Strict = RetentionMode("strict")
  val All = RetentionMode("all")
  protected case class RetentionMode(retentionMode: String) extends super.Val()
  implicit def convert(value: Value): RetentionMode =
    value.asInstanceOf[RetentionMode]
}
