package com.databricks.labs.automl.ensemble.validation

import com.databricks.labs.automl.ensemble.exception.EnsembleValidationException
import com.databricks.labs.automl.ensemble.setting.CoreEnsembleSettings

trait EnsembleSettingsValidator[T <: CoreEnsembleSettings] {

  @throws(classOf[EnsembleValidationException])
  def validate(t: T)

}
