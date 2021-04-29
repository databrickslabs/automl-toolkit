package com.databricks.labs.automl.ensemble.tuner.validate

import com.databricks.labs.automl.ensemble.tuner.exception.TuningException
import com.databricks.labs.automl.params.MainConfig

trait GeneticTunerValidator {

  @throws(classOf[TuningException])
  def validate(mainConfig: MainConfig)

}
