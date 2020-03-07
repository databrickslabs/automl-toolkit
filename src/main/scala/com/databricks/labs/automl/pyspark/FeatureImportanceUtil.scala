package com.databricks.labs.automl.pyspark

import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import com.fasterxml.jackson.databind.ObjectMapper
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig}
import org.apache.spark.sql.SparkSession
import com.databricks.labs.automl.exploration.FeatureImportances

object FeatureImportanceUtil {
  lazy val objectMapper = new ObjectMapper()

  def runFeatureImportance(modelFamily: String,
                           predictionType: String,
                           configJson: String,
                           df: DataFrame,
                           cutoffType: String,
                           cutoffValue: Float,
                           defaultFlag: String): Unit = {


    val fiConfig  = defaultConfigFlag(defaultFlag,
      configJson,
      modelFamily,
      predictionType)

    val mainConfig = ConfigurationGenerator.generateFeatureImportanceConfig(fiConfig)
    val fImportances = FeatureImportances(df, mainConfig, cutoffType, cutoffValue).generateFeatureImportances()

    //create temp importances df and top fields to get them later in python
    fImportances.importances.createOrReplaceTempView("importances")

  }

  def defaultConfigFlag(defaultFlag: String,
                        configJson: String,
                        modelFamily: String,
                        predictionType: String): InstanceConfig = {
    if (defaultFlag == "true"){
      // Generate default config if default flag is true
      val fiConfig = ConfigurationGenerator.generateDefaultConfig(modelFamily, predictionType)
      return fiConfig
    }
    else{
      // Generating config from the map of overrides if default configs aren't being used
      val overrides = jsonToMap(configJson)
      val fiConfig = ConfigurationGenerator.generateConfigFromMap(modelFamily,predictionType,overrides)
      return fiConfig
    }
  }

  def jsonToMap(message: String): Map[String, Any] = {
    objectMapper.registerModule(DefaultScalaModule)
    objectMapper.readValue(message, classOf[Map[String, Any]])
  }

}
