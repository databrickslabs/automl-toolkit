package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, PipelineMlFlowTagKeys, PipelineStatus}

/**
  * @author Jas Bali
  * @since 0.6.1
  * Utility for reporting pipeline progress to MLflow
  */
object PipelineMlFlowProgressReporter {

  private def isMlFlowOn(pipelineId: String): Boolean = {
    AutoMlPipelineMlFlowUtils.getMainConfigByPipelineId(pipelineId).mainConfig.mlFlowLoggingFlag
  }

  private def addProgressToPipelineCache(pipelineId: String, progress: String): Unit = {
    PipelineStateCache
      .addToPipelineCache(
        pipelineId,
        PipelineVars.PIPELINE_STATUS.key, progress)
  }

  private def addProgressToMLflowRun(pipelineId: String, message: String): Unit = {
    if(isMlFlowOn(pipelineId)) {
      AutoMlPipelineMlFlowUtils
        .logTagsToMlFlow(
          pipelineId,
          Map(s"${PipelineMlFlowTagKeys.PIPELINE_STATUS}" -> message
          ))
    }
  }

  def starting(pipelineId: String): Unit = {
    addProgressToPipelineCache(pipelineId, PipelineStatus.PIPELINE_STARTED.key)
    addProgressToMLflowRun(
      pipelineId,
      s"${PipelineStatus.PIPELINE_STARTED} (Building Pipeline from a given configuration)")
  }

  def runningStage(pipelineId: String, stage: String): Unit = {
    addProgressToPipelineCache(pipelineId, PipelineStatus.PIPELINE_RUNNING.key)
    addProgressToMLflowRun(
      pipelineId,
      s"${PipelineStatus.PIPELINE_RUNNING} Stage: $stage")
  }

  def completed(pipelineId: String, totalStagesRan: Int): Unit = {
    addProgressToPipelineCache(pipelineId, PipelineStatus.PIPELINE_COMPLETED.key)
    addProgressToMLflowRun(
      pipelineId,
      s"${PipelineStatus.PIPELINE_COMPLETED}. Total Stages Executed: $totalStagesRan")
  }

  def failed(pipelineId: String, message: String): Unit = {
    addProgressToPipelineCache(pipelineId, PipelineStatus.PIPELINE_FAILED.key)
    addProgressToMLflowRun(
      pipelineId,
      s"${PipelineStatus.PIPELINE_FAILED} with message: $message (Search for pipeline ID $pipelineId in the cluster logs to find more details)")
  }

}
