package com.databricks.labs.automl.exceptions

final case class StringFeatureConversionException(
                                                 private val stringFields: Array[String] = Array.empty,
                                                 private val cause: Throwable = None.orNull)
  extends FeatureConversionException(stringFields, "String")