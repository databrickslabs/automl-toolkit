package com.databricks.labs.automl.exceptions

abstract class FeatureConversionException(private val fields: Array[String] = Array.empty,
                                            private val fieldsType: String,
                                            private val cause: Throwable = None.orNull)
  extends RuntimeException(s"Not all $fieldsType features [[ ${fields.toList} ]] have been converted into vectorizable fields", cause)