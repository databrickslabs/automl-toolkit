package com.databricks.labs.automl.ensemble.exception

final case class EnsembleInvalidSettingsException(private val propertyName: String,
                                                  private val propertyValue: String)
  extends EnsembleValidationException(s"$propertyName value $propertyValue is invalid")

class EnsembleValidationException(private val message: String = "",
                                     private val cause: Exception = None.orNull)
  extends RuntimeException(message, cause)


object EnsembleValidationExceptions {

}
