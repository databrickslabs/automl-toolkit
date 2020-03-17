package com.databricks.labs.automl.pyspark

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.sql.DataFrame
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig}
import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.utils.SparkSessionWrapper
import com.databricks.labs.automl.pipeline.inference.PipelineModelInference

object FamilyRunnerUtil extends SparkSessionWrapper {
  lazy val objectMapper = new ObjectMapper()
  def runFamilyRunner(configs:String,
                      predictionType: String,
                      df: DataFrame,
                      tmpPath: String): Unit = {
    import spark.implicits._

    val firstMap = jsonToMap(configs)
    val familyRunnerConfigs = buildArray(firstMap,
      predictionType)
    //run the family runner
    val runner = FamilyRunner(df, familyRunnerConfigs).executeWithPipeline()
    runner.familyFinalOutput.modelReportDataFrame.createOrReplaceTempView("modelReportDataFrame")
    runner.familyFinalOutput.generationReportDataFrame.createOrReplaceTempView("generationReportDataFrame")
    runner.bestMlFlowRunId.toSeq.toDF("model_family", "run_id").createOrReplaceTempView("bestMlFlowRunId")

    //Write out all the model to get them later in python

    for(i <- runner.bestMlFlowRunId.keys){
      val savePath = tmpPath + i.asInstanceOf[String]
      runner.bestPipelineModel(i).write.overwrite().
    }

    def runInferencePipeline(mlFlowRunId:String,
                            modelFamily:String,
                            predictionType:String,
                            labelCol:String,
                            configs:Map[String,Any],
                            df:DataFrame): Unit = {

      // TO DO add support for default configs
      // generate the configs
      val configs = ConfigurationGenerator.generateConfigFromMap(modelFamily, predictionType, overrides)
      // get logging config
      val loggingConfig = configs.loggingConfig
      // get pipeline model
      val pipelineModel = PipelineModelInference.getPipelineModelByMlFlowRunId(mlFlowRunId, loggingConfig)
      // run inference on df and pipeline model from mlflow
      val inferenceDataFrame = pipelineModel.transform(df.drop("label")).createOrReplaceTempView("inference_df")
      // store as temp view to get it back in python
      inferenceDataFrame.createOrReplaceTempView("inferenceDF")
    }


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
