package com.databricks.labs.automl.ensemble.setting

import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig, TunerConfig}
import org.apache.spark.sql.DataFrame

trait CoreEnsembleSettings

case class StackingEnsembleSettings(inputData: DataFrame,
                                    weakLearnersConfigs: Array[InstanceConfig],
                                    metaLearnerConfig: Option[InstanceConfig] = None) extends CoreEnsembleSettings
class EnsembleSettingsBuilder {

  private var inputData: DataFrame = _
  private var weakLearnersConfigs: Array[InstanceConfig] = _
  private var metaLearnerConfig: Option[InstanceConfig] = None

  def inputData(inputData: DataFrame): EnsembleSettingsBuilder = {
    this.inputData = inputData
    this
  }

  def weakLearnersConfigs(weakLearnersConfigs: Array[InstanceConfig]): EnsembleSettingsBuilder = {
    this.weakLearnersConfigs = weakLearnersConfigs
    this
  }

  def metaLearnerConfig(metaLearnerConfig: Option[InstanceConfig]): EnsembleSettingsBuilder = {
    this.metaLearnerConfig = metaLearnerConfig
    this
  }

  def build(): StackingEnsembleSettings = {
    StackingEnsembleSettings(
      this.inputData,
      this.weakLearnersConfigs,
      this.metaLearnerConfig)
  }

}

object EnsembleSettingsBuilder {
  def builder(): EnsembleSettingsBuilder = new EnsembleSettingsBuilder()
}
