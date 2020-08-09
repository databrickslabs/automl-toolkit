package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.ensemble.tuner.validate.GeneticTunerValidator
import com.databricks.labs.automl.params.TunerOutput

trait TunerDelegator extends GeneticTunerValidator {

  def tune: TunerOutput

}
