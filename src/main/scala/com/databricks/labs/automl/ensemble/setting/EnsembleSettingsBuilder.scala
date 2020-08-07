package com.databricks.labs.automl.ensemble.setting

import com.databricks.labs.automl.executor.config.{InstanceConfig, TunerConfig}
import org.apache.spark.sql.DataFrame

trait CoreEnsembleSettings

case class StackingEnsembleSettings(inputData: DataFrame,
                                    weakLearnersConfigs: Array[InstanceConfig],
                                    metaLearnerModelType: String,
                                    metaLearnerTuningConfig: Option[TunerConfig]) extends CoreEnsembleSettings
class EnsembleSettingsBuilder {


  private var inputData: DataFrame = _
  private var weakLearnersConfigs: Array[InstanceConfig] = _
  private var metaLearnerModelType: String = _
  private var metaLearnerTuningConfig: Option[TunerConfig] = None


  def inputData(inputData: DataFrame): EnsembleSettingsBuilder = {
    this.inputData = inputData
    this
  }

  def weakLearnersConfigs(weakLearnersConfigs: Array[InstanceConfig]): EnsembleSettingsBuilder = {
    this.weakLearnersConfigs = weakLearnersConfigs
    this
  }

  def metaLearnerModelType(metaLearnerModelType: String): EnsembleSettingsBuilder = {
    this.metaLearnerModelType = metaLearnerModelType
    this
  }

  def metaLearnerTuningConfig(metaLearnerTuningConfig: Option[TunerConfig]): EnsembleSettingsBuilder = {
    this.metaLearnerTuningConfig = metaLearnerTuningConfig
    this
  }

  def build(): StackingEnsembleSettings = {
    StackingEnsembleSettings(
      this.inputData,
      this.weakLearnersConfigs,
      this.metaLearnerModelType,
      this.metaLearnerTuningConfig)
  }

}

object EnsembleSettingsBuilder {
  def apply(): EnsembleSettingsBuilder = new EnsembleSettingsBuilder()
}
