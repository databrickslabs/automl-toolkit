package com.databricks.labs.automl.exceptions

case class BooleanFieldFillException(
  private val fieldName: String,
  private val conversionMode: String,
  private val allowableConversionModes: Array[String],
  cause: Throwable = None.orNull
) extends RuntimeException(
      s"The boolean fill type " +
        s"specified: $conversionMode is not in the allowable list of supported models: ${allowableConversionModes
          .mkString(", ")} for field $fieldName",
      cause
    )
