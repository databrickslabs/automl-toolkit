package com.databricks.labs.automl.feature

import com.databricks.labs.automl.exceptions.ModelingTypeException

trait FeatureInteractionBase {
  import ModelingType._
  import FieldEncodingType._

  private final val allowableModelTypes = Array("classifier", "regressor")
  private final val allowableFieldTypes = Array("nominal", "continuous")

  final val AGGREGATE_COLUMN: String = "totalCount"
  final val COUNT_COLUMN: String = "count"
  final val RATIO_COLUMN: String = "labelRatio"
  final val TOTAL_RATIO_COLUMN: String = "totalRatio"
  final val ENTROPY_COLUMN: String = "entropy"
  final val FIELD_ENTROPY_COLUMN: String = "fieldEntropy"
  final val QUANTILE_THRESHOLD: Double = 0.5
  final val QUANTILE_PRECISION: Double = 0.95
  final val VARIANCE_STATISTIC: String = "stddev"

  protected[feature] def getModelType(
    modelingType: String
  ): ModelingType.Value = {
    modelingType match {
      case "regressor"  => Regressor
      case "classifier" => Classifier
      case _            => throw ModelingTypeException(modelingType, allowableModelTypes)
    }
  }

  protected[feature] def getFieldType(
    fieldType: String
  ): FieldEncodingType.Value = {
    fieldType match {
      case "nominal"    => Nominal
      case "continuous" => Continuous
      case _            => throw ModelingTypeException(fieldType, allowableFieldTypes)
    }
  }

}

case class VarianceData(labelValue: Double, variance: Double)
case class EntropyData(labelValue: Double, entropy: Double)
case class InteractionPayload(left: String, right: String, outputName: String)

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
