package com.databricks.labs.automl.pyspark

import com.databricks.labs.automl.executor.config.{
  ConfigurationGenerator,
  InstanceConfig
}
import com.databricks.labs.automl.exploration.FeatureImportances
import com.databricks.labs.automl.pyspark.utils.Utils
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.sql.DataFrame

object FeatureImportanceUtil {
  lazy val objectMapper = new ObjectMapper()

  def runFeatureImportance(modelFamily: String,
                           predictionType: String,
                           configJson: String,
                           df: DataFrame,
                           cutoffType: String,
                           cutoffValue: Float,
                           defaultFlag: String): Unit = {

    val fiConfig =
      defaultConfigFlag(defaultFlag, configJson, modelFamily, predictionType)

    val mainConfig =
      ConfigurationGenerator.generateFeatureImportanceConfig(fiConfig)
    val fImportances =
      FeatureImportances(df, mainConfig, cutoffType, cutoffValue)
        .generateFeatureImportances()

    //create temp importances df and top fields to get them later in python
    fImportances.importances.createOrReplaceTempView("importances")

  }

  def defaultConfigFlag(defaultFlag: String,
                        configJson: String,
                        modelFamily: String,
                        predictionType: String): InstanceConfig = {
    if (defaultFlag == "true") {
      // Generate default config if default flag is true
      ConfigurationGenerator.generateDefaultConfig(modelFamily, predictionType)
    } else {
      // Generating config from the map of overrides if default configs aren't being used
      val overrides = Utils.cleansNestedTypes(jsonToMap(configJson))
      ConfigurationGenerator.generateConfigFromMap(
        modelFamily,
        predictionType,
        overrides
      )
    }
  }

  def jsonToMap(message: String): Map[String, Any] = {
    objectMapper.registerModule(DefaultScalaModule)
    objectMapper.readValue(message, classOf[Map[String, Any]])
  }

}
