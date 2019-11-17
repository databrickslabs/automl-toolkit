package com.databricks.labs.automl

import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.pipeline.inference.PipelineModelInference
import com.databricks.labs.automl.params.MLFlowConfig
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Runner extends App {

  val spark = SparkSession.builder()
    .master("local")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")
//  spark.conf.set("spark.databricks.service.client.checkDeps", false)
  import spark.implicits._
  val sc = spark.sparkContext
  sc.addJar("/Users/danieltomes/Dev/gitProjects/Databricks---AutoML--providentia/target/scala-2.11/automatedml_2.11-0.6.1.jar")

  val df = spark.table("tke_features.cost_features")
    .withColumn("cancellation_term", when('cancellation_term.isNull, "NA").otherwise('cancellation_term))
    .withColumn("TARGET", lit("HOLD"))
    .drop("ch_start_snap_yr")
    .limit(200)
    .na.fill("NA")

  val engOvrCgfs = Map(
    "labelCol" -> "TARGET",
    "fieldsToIgnoreInVector" -> Array("UnitID"),
    "scoringMetric" -> "f1",
    "naFillFlag" -> true,
    "varianceFilterFlag" -> true,
    "outlierFilterFlag" -> false,
    "pearsonFilterFlag" -> false,
    "covarianceFilterFlag" -> false,
    "oneHotEncodeFlag" -> false,
    "scalingFlag" -> false,
    "dataPrepCachingFlag" -> false,
    "mlFlowLoggingFlag" -> false,
    "mlFlowLogArtifactsFlag" -> false
  )

  val engConfig = ConfigurationGenerator.generateConfigFromMap("XGBoost", "classifier", engOvrCgfs)
  val featureEngPipelineModel = FamilyRunner(df, Array(engConfig)).generateFeatureEngineeredPipeline(verbose=true)("XGBoost")
  val featurizedDF = featureEngPipelineModel.transform(df)
  featurizedDF.show()
}
