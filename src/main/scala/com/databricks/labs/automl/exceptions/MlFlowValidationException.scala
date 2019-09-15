package com.databricks.labs.automl.exceptions


final case class MlFlowValidationException(private val message: String = "",
                                   private val cause: Throwable = None.orNull) extends RuntimeException(message, cause)


