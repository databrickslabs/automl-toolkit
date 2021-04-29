package com.databricks.labs.automl.ensemble.exception

final case class EnsembleInvalidSettingsException(private val propertyName: String,
                                                  private val propertyValue: String)
  extends EnsembleValidationException(s"$propertyName value $propertyValue is invalid")

class EnsembleValidationException(private val message: String = "",
                                  private val cause: Exception = None.orNull)
  extends RuntimeException(message, cause)


object EnsembleValidationExceptions {

  val TRAIN_PORTION_EXCEPTION: EnsembleValidationException =
    new EnsembleValidationException("Train Split Config must be same for all weak learner configs")

  val KSAMPLE_NOT_SUPPORTED: EnsembleValidationException =
    new EnsembleValidationException(
      "Ksample split type isn't supported for stacking ensemble",
      new UnsupportedOperationException)

}
