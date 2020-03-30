package com.databricks.labs.automl.exceptions

final case class DateFeatureConversionException(
                                         private val dateFields: Array[String] = Array.empty,
                                         private val cause: Throwable = None.orNull)
  extends FeatureConversionException(dateFields, "Date")