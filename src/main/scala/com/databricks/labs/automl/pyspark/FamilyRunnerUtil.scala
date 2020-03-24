package com.databricks.labs.automl.pyspark

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.sql.DataFrame
import com.databricks.labs.automl.executor.config.{
  ConfigurationGenerator,
  InstanceConfig
}
import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.pyspark.utils.Utils
import com.databricks.labs.automl.utils.SparkSessionWrapper
import com.databricks.labs.automl.pipeline.inference.PipelineModelInference
import org.apache.spark.ml.PipelineModel

object FamilyRunnerUtil extends SparkSessionWrapper {
  lazy val objectMapper = new ObjectMapper()
  def runFamilyRunner(df: DataFrame,
                      configs: String,
                      predictionType: String): Unit = {
    import spark.implicits._

    val firstMap = jsonToMap(configs)
    val familyRunnerConfigs = buildArray(firstMap, predictionType)
    //run the family runner
    val runner = FamilyRunner(df, familyRunnerConfigs).executeWithPipeline()
    runner.familyFinalOutput.modelReportDataFrame
      .createOrReplaceTempView("modelReportDataFrame")
    runner.familyFinalOutput.generationReportDataFrame
      .createOrReplaceTempView("generationReportDataFrame")
    runner.bestMlFlowRunId.toSeq
      .toDF("model_family", "run_id")
      .createOrReplaceTempView("bestMlFlowRunId")
  }

  def cleansNestedTypes(valuesMap: Map[String, Any]): Map[String, Any] = {
    val cleanMap: scala.collection.mutable.Map[String, Any] =
      scala.collection.mutable.Map()
    if (valuesMap.contains("fieldsToIgnoreInVector")) {
      cleanMap("fieldsToIgnoreInVector") = valuesMap("fieldsToIgnoreInVector")
        .asInstanceOf[List[String]]
        .toArray
    }
    if (valuesMap.contains("outlierFieldsToIgnore")) {
      cleanMap("outlierFieldsToIgnore") = valuesMap("outlierFieldsToIgnore")
        .asInstanceOf[List[String]]
        .toArray
    }
    if (valuesMap.contains("numericBoundaries")) {
      cleanMap("numericBoundaries") = valuesMap("numericBoundaries")
        .asInstanceOf[Map[String, List[Any]]]
        .flatMap({
          case (k, v) => {
            Map(k -> Tuple2(v.head.toString.toDouble, v(1).toString.toDouble))
          }
        })
    }
    cleanMap.toMap
  }

  def runMlFlowInference(mlFlowRunId: String,
                         modelFamily: String,
                         predictionType: String,
                         labelCol: String,
                         configs: String,
                         df: DataFrame): Unit = {

    // TO DO add support for default configs
    // generate the configs
    val familyRunnerConfigs = ConfigurationGenerator.generateConfigFromMap(
      modelFamily,
      predictionType,
      jsonToMap(configs)
    )
    // get logging config
    val loggingConfig = familyRunnerConfigs.loggingConfig
    // get pipeline model
    val pipelineModel = PipelineModelInference.getPipelineModelByMlFlowRunId(
      mlFlowRunId,
      loggingConfig
    )
    // run inference on df and pipeline model from mlflow
    pipelineModel
      .transform(df.drop(labelCol))
      .createOrReplaceTempView("inferenceDF")
  }

  def runPathInference(path: String, dataframe: DataFrame): Unit = {
    // Read in the Pipeline
    PipelineModel
      .load(path)
      .transform(dataframe)
      .createOrReplaceTempView(viewName = "pathInferenceDF")
  }

  def runFeatureEngPipeline(df: DataFrame,
                            modelFamily: String,
                            predictionType: String,
                            configs: String): Unit = {
    import spark.implicits._

    val firstMap = jsonToMap(configs)
    val familyRunnerConfigs = buildArray(firstMap, predictionType)
    //run the family runner
    val featureEngPipelineModel = FamilyRunner(df, familyRunnerConfigs)
      .generateFeatureEngineeredPipeline(verbose = true)(modelFamily)
    featureEngPipelineModel
      .transform(df)
      .createOrReplaceTempView(viewName = "featEngDf")
  }

  def buildArray(configs: Map[String, Any],
                 predictionType: String): Array[InstanceConfig] = {

    configs
      .asInstanceOf[Map[String, Map[String, Any]]]
      .map({
        case (key, rawValuesMap) => {
          val valuesMap: Map[String, Any] = rawValuesMap ++ Utils
            .cleansNestedTypes(rawValuesMap)
          ConfigurationGenerator
            .generateConfigFromMap(key, predictionType, valuesMap)
        }
      })
      .toArray
  }

  def jsonToMap(message: String): Map[String, Any] = {
    objectMapper.registerModule(DefaultScalaModule)
    objectMapper.readValue(message, classOf[Map[String, Any]])
  }

}
