package com.databricks.labs.automl.exceptions

final case class ModelingTypeException(
  private val modelType: String,
  private val allowableModelTypes: Array[String],
  cause: Throwable = None.orNull
) extends RuntimeException(
      s"The model type " +
        s"specified: $modelType is not in the allowable list of supported models: ${allowableModelTypes
          .mkString(", ")}",
      cause
    )
