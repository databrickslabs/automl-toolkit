package com.databricks.labs.automl.pipeline

import org.apache.log4j.Logger
import org.apache.spark.ml.param.{BooleanParam, Param, Params}
import org.apache.spark.sql.Dataset

/**
  * Base trait for setting/accessing debug flags. Meant to be extended by all pipeline stages,
  * which inherit pipeline staging logging by default
  */
trait HasDebug extends Params {

  private val logger: Logger = Logger.getLogger(this.getClass)

  final val isDebugEnabled: BooleanParam = new BooleanParam(this, "isDebugEnabled", "Debug option flag")

  def setDebugEnabled(value: Boolean): this.type = set(isDebugEnabled, value)

  def getDebugEnabled: Boolean = $(isDebugEnabled)

  def logTransformation(inputDataset: Dataset[_], outputDataset: Dataset[_]): Unit = {
    if(getDebugEnabled) {
      // Keeping this INFO level, since debug level can easily pollute this important block of debug information
      logger.info( "\n \n" +
        "=== AutoML Custom Transformers log ==> \n \n" +
        s"Stage: ${this.getClass} \n" +
        s"Stage Params: ${paramsAsString(this.params)} \n " +
        s"Input dataset count: ${inputDataset.count()} \n " +
        s"Output dataset count: ${outputDataset.count()} \n " +
        s"Input dataset schema: ${inputDataset.schema.treeString} \n " +
        s"Output dataset schama: ${outputDataset.schema.treeString} " + "\n" +
        "=== End of AutoML Custom Transformer log <==" + "\n")
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
