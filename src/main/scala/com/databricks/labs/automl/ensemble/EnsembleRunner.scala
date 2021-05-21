package com.databricks.labs.automl.ensemble

import com.databricks.labs.automl.ensemble.impl.StackingEnsembleLearner
import com.databricks.labs.automl.ensemble.setting.StackingEnsembleSettings
import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.TunerConfig
import com.databricks.labs.automl.params.FamilyFinalOutput
import org.apache.spark.ml.PipelineModel

object EnsembleRunner {

  def stacking(stackingEnsembleSettings: StackingEnsembleSettings): Option[EnsembleReturnType] = {
    new StackingEnsembleLearner().run(stackingEnsembleSettings)
  }

}
