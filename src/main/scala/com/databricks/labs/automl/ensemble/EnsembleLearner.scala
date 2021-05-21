package com.databricks.labs.automl.ensemble

import com.databricks.labs.automl.ensemble.setting.CoreEnsembleSettings
import com.databricks.labs.automl.ensemble.validation.EnsembleSettingsValidator
import com.databricks.labs.automl.params.FamilyFinalOutput
import org.apache.spark.ml.PipelineModel

case class EnsembleReturnType(bestEnsembleModel: PipelineModel,
                              bestEnsembleMlFlowRunId: String,
                              weakLearners: FamilyFinalOutput,
                              metaLearner: FamilyFinalOutput)

abstract class EnsembleLearner[T <: CoreEnsembleSettings] extends EnsembleSettingsValidator[T] {

  protected def execute(t: T): Option[EnsembleReturnType]

  def run(t: T): Option[EnsembleReturnType] = {
    validate(t)
    execute(t)
  }

}
