package com.databricks.labs.automl.executor.config

import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class ConfigurationTest extends AbstractUnitSpec {

  it should "serialize and deserialize the config properly" in {

    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()

    val configurationOverrides = Map(
      "labelCol" -> "income",
      "tunerParallelism" -> 6,
      "tunerKFold" -> 1,
      "featureInteractionFlag" -> false,
      "scoringMetric" -> "areaUnderROC",
      "featureInteractionRetentionMode" -> "all",
      "inferenceConfigSaveLocation" -> "dbfs:/test/test/test",
      "mlFlowModelSaveDirectory" -> "dbfs:/test/test"
    )

    val rfConfig = ConfigurationGenerator.generateConfigFromMap(
      "RandomForest",
      "classifier",
      configurationOverrides
    )

    val rfConfigString =
      ConfigurationGenerator.generatePrettyJsonInstanceConfig(rfConfig)

    val rfConfigToJSON = ConfigurationGenerator.jsonStrToMap(rfConfigString)

    val backToInstanceConfig =
      ConfigurationGenerator.generateInstanceConfigFromJson(rfConfigString)

    val backToJSON = ConfigurationGenerator.jsonStrToMap(
      ConfigurationGenerator
        .generatePrettyJsonInstanceConfig(backToInstanceConfig)
    )

    assert(backToJSON == rfConfigToJSON)

  }

}
