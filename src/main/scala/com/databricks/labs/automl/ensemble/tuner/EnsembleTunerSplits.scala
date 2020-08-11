package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.ensemble.setting.StackingEnsembleSettings
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfigValidation}
import com.databricks.labs.automl.model.tools.structures.{TrainSplitReferences, TrainTestData}

class EnsembleTunerSplits {

  private var metaLearnerSplits: Option[Array[TrainSplitReferences]] = None
  private var weakLearnerSplits: Option[Array[TrainSplitReferences]] = None

  /*
    To get splits for weak learners( for k = 1):
      1) Get splits of meta learners
      2) use train data from step 1 as an input to calculate splits for weak learners
      3) Override test data for step 2 from step 1

    To get splits for weak learners( for k > 1):
      1) Get splits of meta learners
      2) use train data from step 1 (first kfold split?) as an input to calculate splits for weak learners
      3) Don't override test data for step 2
   */
  def getWeakLearnersSplits(stackingEnsembleSettings: StackingEnsembleSettings): Array[TrainSplitReferences] = {
    if(weakLearnerSplits.isEmpty) {
      val weakConfig = ConfigurationGenerator.generateMainConfig(stackingEnsembleSettings.weakLearnersConfigs(0))
      val metaSplits = getMetaLearnersSplits(stackingEnsembleSettings)
      if(weakConfig.geneticConfig.kFold > 1) {
        val weakSplits = TunerUtils.buildSplitTrainTestData(weakConfig, metaSplits(0).data.train)
        weakLearnerSplits = Some(weakSplits)
      } else {
        val weakSplits = TunerUtils
          .buildSplitTrainTestData(weakConfig, metaSplits(0).data.train)
          .map(item =>
            TrainSplitReferences(
              item.kIndex,
              TrainTestData(item.data.train, metaSplits(0).data.test),
              item.paths
            )
          )
        weakLearnerSplits = Some(weakSplits)
      }
    }
    weakLearnerSplits.get
  }

  def getMetaLearnersSplits(stackingEnsembleSettings: StackingEnsembleSettings): Array[TrainSplitReferences] = {
    if(metaLearnerSplits.isEmpty) {
      val metaConfig = ConfigurationGenerator.generateMainConfig(stackingEnsembleSettings.metaLearnerConfig)
      val metaSplits =  TunerUtils.buildSplitTrainTestData(metaConfig, stackingEnsembleSettings.inputData)
      metaLearnerSplits = Some(metaSplits)
    }
    metaLearnerSplits.get
  }

}
object EnsembleTunerSplits {
  def apply(): EnsembleTunerSplits = new EnsembleTunerSplits()
}
