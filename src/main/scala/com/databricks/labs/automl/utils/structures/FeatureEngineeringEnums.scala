package com.databricks.labs.automl.utils.structures

object FeatureEngineeringEnums extends Enumeration {

  type FeatureEngineeringEnums = FeatureEngineeringConstants

  val MIN = FeatureEngineeringConstants("min")
  val MAX = FeatureEngineeringConstants("max")
  val COUNT_COL = FeatureEngineeringConstants("count")

  case class FeatureEngineeringConstants(value: String) extends Val
}

object FeatureEngineeringAllowables extends Enumeration {

  type FeatureEngineeringAllowables = FeatureEngineeringAllowableConstants

  val ALLOWED_CATEGORICAL_FILL_MODES = FeatureEngineeringAllowableConstants(
    Array("min", "max")
  )

  case class FeatureEngineeringAllowableConstants(values: Array[String])
      extends Val

}

object ArrayGeneratorMode extends Enumeration {
  val ASC = ArrayMode("ascending")
  val DESC = ArrayMode("descending")
  val RAND = ArrayMode("random")
  protected case class ArrayMode(arrayMode: String) extends super.Val()
  implicit def convert(value: Value): ArrayMode = value.asInstanceOf[ArrayMode]
}
