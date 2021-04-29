package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.ensemble.setting.StackingEnsembleSettings
import com.databricks.labs.automl.pipeline.PipelineInternalUtils

private[ensemble] class WeakLearnersFamilyRunner
    extends EnsembleFamilyRunnerLike[EnsembleFamilyRunnerLikeReturnType, Any] {

  override def execute(stackingEnsembleSettings: StackingEnsembleSettings,
                       b: Option[Any]): EnsembleFamilyRunnerLikeReturnType = {
    val configs = stackingEnsembleSettings.weakLearnersConfigs
    // convert configs into main configs
    val mainConfigs = getMainConfigs(configs)
    // Generate PipelineModel to get features column for all weak learner models
    val fePipelineModels = getFePipelineModels(stackingEnsembleSettings.inputData, mainConfigs)

    // Combine all weak learners FE PipelineModels
    val weakLearnersFePipelineModel = PipelineInternalUtils.mergePipelineModels(fePipelineModels.map(_.pipelineModel))
    // Get df with all features columns for weak learners
    val feDfWithAllModelFeatures = weakLearnersFePipelineModel.transform(stackingEnsembleSettings.inputData)
    // Get Train-Test splits to be reused for all weak learners' tunings
    val ensembleTunerSplits = EnsembleTunerSplits()
    val weakLearnersTrainTestSplits = ensembleTunerSplits
      .getMetaLearnersSplits(Some(StackingEnsembleSettings(
        feDfWithAllModelFeatures,
        stackingEnsembleSettings.weakLearnersConfigs,
        stackingEnsembleSettings.metaLearnerConfig)))

    // Run Tuning for all weak learner models
    val (outputBuffer, pipelineConfigMap) = runTuningAndGetOutput(
      fePipelineModels,
      mainConfigs,
      configs,
      feDfWithAllModelFeatures,
      weakLearnersTrainTestSplits)

    EnsembleFamilyRunnerLikeReturnType(
      withPipelineInferenceModel(
        stackingEnsembleSettings.inputData,
        unifyFamilyOutput(outputBuffer.toArray),
        configs,
        pipelineConfigMap.toMap
      ),
      Some(ensembleTunerSplits))
  }
}

private[ensemble] object WeakLearnersFamilyRunner {
  def apply(): WeakLearnersFamilyRunner = new WeakLearnersFamilyRunner()
}