package com.databricks.labs.automl.model.tools.split

import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.log4j.{Level, Logger}

import scala.collection.JavaConverters._

object PerformanceSettings extends SparkSessionWrapper {

  @transient private val logger: Logger = Logger.getLogger(this.getClass)

  final val environmentVars: Map[String, String] = System.getenv().asScala.toMap

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

  private lazy val preCalcParTasks: Int =
    scala.math.floor(totalCores / coresPerTask).toInt
  lazy val parTasks: Int = if (preCalcParTasks < 1) 1 else preCalcParTasks

  def envString: String =
    s"coresPerWorker: $coresPerWorker \n" +
      s"numberOfWorkerNodes: $numberOfWorkerNodes \n " +
      s"totalCores: $totalCores \n " +
      s"coresPerTask: $coresPerTask \n " +
      s"preCalcParTasks: $preCalcParTasks \n " +
      s"parTasks: $parTasks"

  def xgbWorkers(parallelism: Int): Int = {
    //DEBUG
    logger.log(Level.DEBUG, envString)

    try {
      environmentVars("num_workers").toString.toInt
    } catch {
      case e: java.util.NoSuchElementException =>
        scala.math.floor(totalCores / coresPerTask / parallelism).toInt
    }
  }

  def optimalJVMModelPartitions(parallelism: Int): Int = {
    //DEBUG
    logger.log(Level.DEBUG, envString)

    scala.math.floor(parTasks / (parallelism / 2)).toInt
  }

}
