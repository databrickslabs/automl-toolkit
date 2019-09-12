package com.databricks.labs.automl.pipeline

import org.apache.log4j.Logger
import org.apache.spark.ml.param.{BooleanParam, Param, Params}
import org.apache.spark.sql.Dataset

/**
  * Base trait for setting/accessing debug flags. Meant to be extended by all pipeline stages,
  * which inherit pipeline staging logging by default
  */
trait HasDebug extends Params {

  @transient private val logger: Logger = Logger.getLogger(this.getClass)

  final val isDebugEnabled: BooleanParam = new BooleanParam(this, "isDebugEnabled", "Debug option flag")

  def setDebugEnabled(value: Boolean): this.type = set(isDebugEnabled, value)

  def getDebugEnabled: Boolean = $(isDebugEnabled)

  def logTransformation(inputDataset: Dataset[_],
                        outputDataset: Dataset[_],
                        stageExecutionTime: Long): Unit = {
    if(getDebugEnabled) {
      // Keeping this INFO level, since debug level can easily pollute this important block of debug information
      val logStrng = s"\n \n" +
        s"=== AutoML Pipeline Stage: ${this.getClass} log ==> \n" +
        s"Stage Name: ${this.uid} \n" +
        s"Total Stage Execution time: ${stageExecutionTime/1000} seconds \n" +
        s"Stage Params: ${paramsAsString(this.params)} \n " +
        s"Input dataset count: ${inputDataset.count()} \n " +
        s"Output dataset count: ${outputDataset.count()} \n " +
        s"Input dataset schema: ${inputDataset.schema.treeString} \n " +
        s"Output dataset schama: ${outputDataset.schema.treeString} " + "\n" +
        s"=== End of ${this.getClass} Pipeline Stage log <==" + "\n"
      println(logStrng)
      logger.info(logStrng)
    }
  }

  private def paramsAsString(params: Array[Param[_]]): String = {
    params.map { param =>
      s"\t${param.toString()}: ${paramValueAsString(this.extractParamMap().get(param).get)}"
    }.mkString("{\n", ",\n", "\n}")
  }

  private def paramValueAsString(value: Any): Any = {
    value match {
      case v: Array[String] =>
        v.asInstanceOf[Array[String]].mkString(", ")
      case _ => value
    }
  }
}
