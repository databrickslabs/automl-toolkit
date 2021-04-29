package com.databricks.labs.automl.model.tools.split

import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

import scala.collection.JavaConverters._

object PerformanceSettings extends SparkSessionWrapper {

  @transient private val logger: Logger = Logger.getLogger(this.getClass)

  final val environmentVars: Map[String, String] = System.getenv().asScala.toMap

  lazy val coresPerWorker: Int = if(!sc.isLocal) {
    sc
      .parallelize("1", 1)
      .map(_ => java.lang.Runtime.getRuntime.availableProcessors)
      .collect()(0)
  } else {
    4
  }

  lazy val numberOfWorkerNodes
    : Int = if(!sc.isLocal) {
    sc.statusTracker.getExecutorInfos.length - 1
  } else {
    1
  }

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

  @throws(classOf[IllegalArgumentException])
  def xgbWorkers(parallelism: Int): Int = {
    //DEBUG
    logger.log(Level.DEBUG, envString)

    try {
      environmentVars("num_workers").toString.toInt
    } catch {
      case e: java.util.NoSuchElementException =>
        val workerCount =
          scala.math.floor(totalCores / coresPerTask / parallelism).toInt
        if(!SparkSession.builder().getOrCreate().sparkContext.isLocal) {
          require(
            workerCount >= 1,
            s"XGBoost requires at least one core per XGB worker. " +
              s"Current configuration is not compatible with XGBoost. Consider increasing cluster size or " +
              s"decreasing parallelism or lowering spark.task.cpus. The XGBWorker count is derived: " +
              s"floor(total Cluster Cores / spark.task.cpus / parallelism).toInt. This number must be >= 1. \n " +
              s"XGB numWorkers == ${workerCount} \n " +
              s"Total Cluster Cores == ${totalCores} \n " +
              s"spark.task.cpu == ${coresPerTask} == nThread \n" +
              s"Parallelism == ${parallelism}"
          )
        }
        workerCount
    }
  }

  // TODO evaluating a theory
//  def optimalJVMModelPartitions(parallelism: Int): Int = {
//    //DEBUG
//    logger.log(Level.DEBUG, envString)
//    val jvmParts = scala.math.floor(parTasks / (parallelism / 2)).toInt
//    val warnMessage = s"WARNING: JVM Model partitions < 10. Consider a larger" +
//      s"cluster or reducing Parallelism. JVM Model Parallelism is calculated: floor(parTasks / (parallelism / 2)). \n " +
//      s"JVM Parallelism: ${jvmParts} \n " +
//      s"parTasks: ${parTasks} \n " +
//      s"Parallelism: ${parallelism}"
//    if (jvmParts < 10) logger.log(Level.WARN, warnMessage)
//    println(warnMessage)
//    jvmParts
//  }

  def optimalJVMModelPartitions(parallelism: Int): Int = {

    logger.log(Level.DEBUG, envString)
    val jvmPartsPre = scala.math.floor(parTasks / (parallelism / 2)).toInt

    val jvmParts = math.max(jvmPartsPre, numberOfWorkerNodes * 2)

    val warnMessage = s"WARNING: JVM Model partitions < 10. Consider a larger" +
      s"cluster or reducing Parallelism. JVM Model Parallelism is calculated: floor(parTasks / (parallelism / 2)). \n " +
      s"JVM Parallelism: ${jvmParts} \n " +
      s"parTasks: ${parTasks} \n " +
      s"Parallelism: ${parallelism}"
    if (jvmParts < 10) {
      logger.log(Level.WARN, warnMessage)
      println(warnMessage)
    }
    jvmParts
  }

}
