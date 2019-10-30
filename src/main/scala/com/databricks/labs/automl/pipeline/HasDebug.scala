package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.params.{MLFlowConfig, MainConfig}
import com.databricks.labs.automl.pipeline.PipelineVars.PIPELINE_LABEL_REFACTOR_NEEDED_KEY
import com.databricks.labs.automl.tracking.MLFlowTracker
import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, PipelineMlFlowTagKeys, PipelineStatus}
import org.apache.log4j.Logger
import org.apache.spark.ml.param.{BooleanParam, Param, Params}
import org.apache.spark.sql.Dataset

/**
  * Base trait for setting/accessing debug flags. Meant to be extended by all pipeline stages,
  * which inherit pipeline stage logging by default
  * @author Jas Bali
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
      val stageExecTime = if(stageExecutionTime < 1000) {
        s"$stageExecutionTime ms"
      } else {
        s"${stageExecutionTime.toDouble/1000} seconds"
      }
      val pipelineId = paramValueAsString(this.extractParamMap().get(this.getParam("pipelineId")).get)
        .asInstanceOf[String]
      val mainCOnfig = PipelineStateCache
        .getFromPipelineByIdAndKey(
          pipelineId,
          PipelineVars.MAIN_CONFIG.key)
        .asInstanceOf[MainConfig]
      //Log Dfs counts
      val countLog = if(mainCOnfig.dataPrepCachingFlag) {
        s"Input dataset count: ${inputDataset.count()} \n " +
        s"Output dataset count: ${outputDataset.count()} \n "
      } else {
        ""
      }
      //TODO: Log Schema flag (required when schemas are large and need to be turned off from log)
      val logStrng = s"\n \n" +
        s"=== AutoML Pipeline Stage: ${this.getClass} log ==> \n" +
        s"Stage Name: ${this.uid} \n" +
        s"Total Stage Execution time: $stageExecTime \n" +
        s"Stage Params: ${paramsAsString(this.params)} \n " +
        s"$countLog" +
        s"Input dataset schema: ${inputDataset.schema.treeString} \n " +
        s"Output dataset schema: ${outputDataset.schema.treeString} " + "\n" +
        s"=== End of ${this.getClass} Pipeline Stage log <==" + "\n"
      // Keeping this INFO level, since debug level can easily pollute this important block of debug information
      println(logStrng)
      logger.info(logStrng)
      //Log this stage to MLFlow with useful information
      val pipelineStatus = try {
        PipelineStateCache
          .getFromPipelineByIdAndKey(
            pipelineId,
            PipelineVars.PIPELINE_STATUS.key)
          .asInstanceOf[String]
      } catch {
        case ex: Exception => PipelineStatus.PIPELINE_FAILED.key
      }
      val isTrain = !pipelineStatus.equals(PipelineStatus.PIPELINE_COMPLETED.key) &&
                    !pipelineStatus.equals(PipelineStatus.PIPELINE_FAILED.key)
      if(!inputDataset.sparkSession.sparkContext.isLocal && isTrain) {
        AutoMlPipelineMlFlowUtils
          .logTagsToMlFlow(pipelineId, Map(s"pipeline_stage_${this.getClass.getName}" -> logStrng))
        PipelineMlFlowProgressReporter.runningStage(pipelineId, this.getClass.getName)
      }
    }
  }

  private def paramsAsString(params: Array[Param[_]]): String = {
    params.map { param =>
      s"\t${param.name}: ${paramValueAsString(this.extractParamMap().get(param).get)}"
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
