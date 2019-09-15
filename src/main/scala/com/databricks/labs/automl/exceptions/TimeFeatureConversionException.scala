package com.databricks.labs.automl.exceptions

final case class TimeFeatureConversionException(
                                                   private val timeFields: Array[String] = Array.empty,
                                                   private val cause: Throwable = None.orNull)
  extends FeatureConversionException(timeFields, "Time")