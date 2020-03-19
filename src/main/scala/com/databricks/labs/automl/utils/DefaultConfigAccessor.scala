package com.databricks.labs.automl.utils

import com.databricks.labs.automl.executor.config.{
  ConfigurationDefaults,
  InstanceConfig
}
import com.databricks.labs.automl.params.{Defaults, MainConfig}

object DefaultConfigAccessor extends Defaults {

  def getMainConfig: MainConfig = _mainConfigDefaults

}

object DefaultInstanceConfigAccessor extends ConfigurationDefaults {

  def getInstanceConfig(modelFamily: String,
                        predictionType: String): InstanceConfig =
    getDefaultConfig(modelFamily, predictionType)

}
