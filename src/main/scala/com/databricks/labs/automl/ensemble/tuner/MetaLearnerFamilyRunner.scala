package com.databricks.labs.automl.ensemble.tuner
import com.databricks.labs.automl.ensemble.setting.StackingEnsembleSettings
import com.databricks.labs.automl.pipeline.PipelineInternalUtils

class MetaLearnerFamilyRunner extends EnsembleFamilyRunnerLike
  [MetaLearnerFamilyRunnerReturnType, EnsembleFamilyRunnerLikeReturnType] {

  override def execute(stackingEnsembleSettings: StackingEnsembleSettings,
                       weakLearnersFamilyRunnerReturnType: Option[EnsembleFamilyRunnerLikeReturnType]): MetaLearnerFamilyRunnerReturnType = {

    val configs = Array(stackingEnsembleSettings.metaLearnerConfig)
    // convert configs into main configs
    val mainConfigs = getMainConfigs(configs)
    // Generate PipelineModel to get features column for all weak learner models
    val fePipelineModels = getFePipelineModels(stackingEnsembleSettings, mainConfigs)
    // Combine all weak learners FE PipelineModels
    val weakLearnersFePipelineModel = PipelineInternalUtils.mergePipelineModels(fePipelineModels.map(_.pipelineModel))
    // Get df with all features columns for weak learners
    val feDfWithAllModelFeatures = weakLearnersFePipelineModel.transform(stackingEnsembleSettings.inputData)
    // Get Train Test Splits for meta learner
    val trainTestSplits = weakLearnersFamilyRunnerReturnType
      .get
      .ensembleTunerSplits
      .get
      .getMetaLearnersSplits(stackingEnsembleSettings)


    // Run Tuning for all weak learner models
    val (outputBuffer, pipelineConfigMap) = runTuningAndGetOutput(
      fePipelineModels,
      mainConfigs,
      configs,
      feDfWithAllModelFeatures,
      trainTestSplits)

    MetaLearnerFamilyRunnerReturnType(
      withPipelineInferenceModel(
        stackingEnsembleSettings.inputData,
        unifyFamilyOutput(outputBuffer.toArray),
        configs,
        pipelineConfigMap.toMap
      ))
  }
}
