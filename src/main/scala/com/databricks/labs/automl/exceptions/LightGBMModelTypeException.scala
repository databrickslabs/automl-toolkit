package com.databricks.labs.automl.exceptions

final case class LightGBMModelTypeException(
  private val modelType: String,
  private val lightGBMType: String,
  lightGBMRegressorTypes: Array[String],
  lightGBMClassifierTypes: Array[String],
  cause: Throwable = None.orNull
) extends RuntimeException(
      s"The model type: [$modelType] and light GBM type: [$lightGBMType] are not a supported combination. Supported " +
        s"types for [$modelType] are: [${modelType match {
          case "regressor"  => lightGBMRegressorTypes.mkString(", ")
          case "classifier" => lightGBMClassifierTypes.mkString(", ")
        }}]",
      cause
    )
