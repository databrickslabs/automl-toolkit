package com.databricks.labs.automl.pyspark

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.sql.DataFrame
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig}
import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.utils.SparkSessionWrapper
import com.databricks.labs.automl.pipeline.inference.PipelineModelInference
import org.apache.spark.ml.PipelineModel

object FamilyRunnerUtil extends SparkSessionWrapper {
  lazy val objectMapper = new ObjectMapper()
  def runFamilyRunner(configs:String,
                      predictionType: String,
                      df: DataFrame): Unit = {
    import spark.implicits._

    val firstMap = jsonToMap(configs)
    val familyRunnerConfigs = buildArray(firstMap,predictionType)
    //run the family runner
    val runner = FamilyRunner(df, familyRunnerConfigs).executeWithPipeline()
    runner.familyFinalOutput.modelReportDataFrame.createOrReplaceTempView("modelReportDataFrame")
    runner.familyFinalOutput.generationReportDataFrame.createOrReplaceTempView("generationReportDataFrame")
    runner.bestMlFlowRunId.toSeq.toDF("model_family", "run_id").createOrReplaceTempView("bestMlFlowRunId")
  }

  def runMlFlowInference(mlFlowRunId:String,
                            modelFamily:String,
                            predictionType:String,
                            labelCol:String,
                            configs:String,
                            df:DataFrame): Unit = {

      // TO DO add support for default configs
      // generate the configs
      val familyRunnerConfigs = ConfigurationGenerator.generateConfigFromMap(modelFamily, predictionType, jsonToMap(configs))
      // get logging config
      val loggingConfig = familyRunnerConfigs.loggingConfig
      // get pipeline model
      val pipelineModel = PipelineModelInference.getPipelineModelByMlFlowRunId(mlFlowRunId, loggingConfig)
      // run inference on df and pipeline model from mlflow
      pipelineModel.transform(df.drop(labelCol)).createOrReplaceTempView("pathInferenceDF")
    }

  def runPathInference(path: String,
                       dataframe: DataFrame): Unit = {
    // Read in the Pipeline
    PipelineModel.load(path).transform(dataframe).createOrReplaceGlobalTempView(viewName = "pathInferenceDf")
  }



  def buildArray(configs: Map[String, Any],
                 predictionType: String): Array[InstanceConfig] = {

    configs
      .asInstanceOf[Map[String, Map[String, Any]]]
      .map({
        case (key, valuesMap) => {
          ConfigurationGenerator.generateConfigFromMap(key, predictionType, valuesMap)
        }
      })
      .toArray
  }

  def jsonToMap(message: String): Map[String, Any] = {
    objectMapper.registerModule(DefaultScalaModule)
    objectMapper.readValue(message, classOf[Map[String, Any]])
  }

}
