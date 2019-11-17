package com.databricks.labs.automl.pyspark

import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import com.fasterxml.jackson.databind.ObjectMapper
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig}
import org.apache.spark.sql.SparkSession
import com.databricks.labs.automl.AutomationRunner

object AutomationRunnerUtil {
  lazy val objectMapper = new ObjectMapper()

  def runAutomationRunner(modelFamily: String,
                          predictionType: String,
                          configJson: String,
                          df: DataFrame,
                          runnerType: String,
                          defaultFlag: String): Unit = {
    val instanceConfig = defaultConfigFlag(defaultFlag,
      configJson,
      modelFamily,
      predictionType)

    val mainConfig = ConfigurationGenerator.generateMainConfig(instanceConfig)
    if (runnerType == "run"){
      val AutomationRunner = new AutomationRunner(df).setMainConfig(mainConfig).run()
      //create temp view of returns
      AutomationRunner.generationReportDataFrame.createOrReplaceTempView("generationReport")
      AutomationRunner.modelReportDataFrame.createOrReplaceTempView("modelReport")
    }
    else if (runnerType == "confusion"){
      val AutomationRunner = new AutomationRunner(df).setMainConfig(mainConfig).runWithConfusionReport()
      // create temp view of the returns
      AutomationRunner.confusionData.createOrReplaceTempView("confusionData")
      AutomationRunner.predictionData.createOrReplaceTempView("predictionData")
      AutomationRunner.generationReportDataFrame.createOrReplaceTempView("generationReport")
      AutomationRunner.modelReportDataFrame.createOrReplaceTempView("modelReport")

    }
    else if (runnerType == "prediction"){
      val AutomationRunner = new AutomationRunner(df).setMainConfig(mainConfig).runWithPrediction()
      //create temp view of the returns
      AutomationRunner.dataWithPredictions.createOrReplaceTempView("dataWithPredictions")
      AutomationRunner.generationReportDataFrame.createOrReplaceTempView("generationReport")
      AutomationRunner.modelReportDataFrame.createOrReplaceTempView("modelReportData")
    }
  }

  def defaultConfigFlag(defaultFlag: String,
                        configJson: String,
                        modelFamily: String,
                        predictionType: String): InstanceConfig = {
    if (defaultFlag == "true"){
      // Generate default config if default flag is true
      val instanceConfig = ConfigurationGenerator.generateDefaultConfig(modelFamily, predictionType)
      return instanceConfig
    }
    else{
      // Generating config from the map of overrides if default configs aren't being used
      val overrides = jsonToMap(configJson)
      val instanceConfig = ConfigurationGenerator.generateConfigFromMap(modelFamily,predictionType,overrides)
      return instanceConfig
    }
  }

  def jsonToMap(message: String): Map[String, Any] = {
    objectMapper.registerModule(DefaultScalaModule)
    objectMapper.readValue(message, classOf[Map[String, Any]])
  }

}
