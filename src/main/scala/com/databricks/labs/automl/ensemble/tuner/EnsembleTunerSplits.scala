package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.ensemble.setting.StackingEnsembleSettings
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfigValidation}
import com.databricks.labs.automl.model.tools.structures.{TrainSplitReferences, TrainTestData, TrainTestPaths}

class EnsembleTunerSplits {

  private var metaLearnerSplits: Option[Array[TrainSplitReferences]] = None
  private var weakLearnerSplits: Option[Array[TrainSplitReferences]] = None

  def getWeakLearnersSplits(stackingEnsembleSettings: StackingEnsembleSettings): Array[TrainSplitReferences] = {
    if(weakLearnerSplits.isEmpty) {
      val weakConfig = ConfigurationGenerator.generateMainConfig(stackingEnsembleSettings.weakLearnersConfigs(0))
      val metaSplits = getMetaLearnersSplits(Some(stackingEnsembleSettings))
      if(weakConfig.geneticConfig.kFold > 1) {
        val weakSplits = metaSplits.map(item => {
          val n = TunerUtils.buildSplitTrainTestData(weakConfig, metaSplits(0).data.train, Some(1))(0)
          TrainSplitReferences(
            item.kIndex,
            TrainTestData(
              n.data.train,
              item.data.test),
            TrainTestPaths(n.paths.train, item.paths.test)
          )
        })
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

  def getMetaLearnersSplits(stackingEnsembleSettings: Option[StackingEnsembleSettings] = None): Array[TrainSplitReferences] = {
    if(metaLearnerSplits.isEmpty) {
      val metaConfig = ConfigurationGenerator.generateMainConfig(stackingEnsembleSettings.get.metaLearnerConfig.get)
      val metaSplits =  TunerUtils.buildSplitTrainTestData(metaConfig, stackingEnsembleSettings.get.inputData)
      metaLearnerSplits = Some(metaSplits)
    }
    metaLearnerSplits.get
  }

}
object EnsembleTunerSplits {
  def apply(): EnsembleTunerSplits = new EnsembleTunerSplits()
}
