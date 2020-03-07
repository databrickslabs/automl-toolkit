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

  val coresPerWorker: Int = sc.parallelize("1", 1).map(_ => java.lang.Runtime.getRuntime.availableProcessors).collect()(0)
  val numberOfWorkerNodes: Int = sc.statusTracker.getExecutorInfos.length - 1
  val totalCores: Int = coresPerWorker * numberOfWorkerNodes
  val coresPerTask: Int = try { spark.conf.get("spark.task.cpus").toInt }
  catch { case e: java.util.NoSuchElementException => 1}

  val environmentVars: Map[String, String] = System.getenv().asScala.toMap
  val parTasks: Int = scala.math.floor(totalCores / coresPerTask).toInt

}
