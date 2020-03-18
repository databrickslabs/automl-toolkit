package com.databricks.labs.automl.utils

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import scala.collection.JavaConverters._

trait SparkSessionWrapper extends Serializable {

  lazy val spark: SparkSession = SparkSession
    .builder()
    .appName("Databricks Automated ML")
    .getOrCreate()

  lazy val sc: SparkContext = SparkContext.getOrCreate()

  lazy val coresPerWorker: Int = sc
    .parallelize("1", 1)
    .map(_ => java.lang.Runtime.getRuntime.availableProcessors)
    .collect()(0)
  lazy val numberOfWorkerNodes
    : Int = sc.statusTracker.getExecutorInfos.length - 1
  lazy val totalCores: Int = coresPerWorker * numberOfWorkerNodes
  lazy val coresPerTask
    : Int = try { spark.conf.get("spark.task.cpus").toInt } catch {
    case e: java.util.NoSuchElementException => 1
  }

  lazy val environmentVars: Map[String, String] = System.getenv().asScala.toMap
  private lazy val preCalcParTasks: Int =
    scala.math.floor(totalCores / coresPerTask).toInt
  lazy val parTasks: Int = if (preCalcParTasks < 1) 1 else preCalcParTasks

}
