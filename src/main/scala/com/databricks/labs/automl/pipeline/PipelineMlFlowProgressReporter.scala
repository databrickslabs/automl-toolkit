package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, PipelineMlFlowTagKeys, PipelineStatus}

/**
  * @author Jas Bali
  * @since 0.6.1
  * Utility for reporting pipeline progress to MLflow
  */
object PipelineMlFlowProgressReporter {

  def starting(pipelineId: String): Unit = {
    AutoMlPipelineMlFlowUtils
      .logTagsToMlFlow(
        pipelineId,
        Map(s"${PipelineMlFlowTagKeys.PIPELINE_STATUS}"
          ->
          s"${PipelineStatus.PIPELINE_STARTED} (Building Pipeline from a given configuration)"
        ))
  }

  def runningStage(pipelineId: String, stage: String): Unit = {
    AutoMlPipelineMlFlowUtils
      .logTagsToMlFlow(
        pipelineId, Map(s"${PipelineMlFlowTagKeys.PIPELINE_STATUS}"
          ->
          s"${PipelineStatus.PIPELINE_RUNNING} Stage: $stage"
        ))
  }

  def completed(pipelineId: String, totalStagesRan: Int): Unit = {
    AutoMlPipelineMlFlowUtils
      .logTagsToMlFlow(
        pipelineId,
        Map(s"${PipelineMlFlowTagKeys.PIPELINE_STATUS}"
          ->
          s"${PipelineStatus.PIPELINE_COMPLETED} Total Stages Executed: $totalStagesRan"
        ))
  }

  def failed(pipelineId: String, message: String): Unit = {
    AutoMlPipelineMlFlowUtils
      .logTagsToMlFlow(
        pipelineId,
        Map(s"${PipelineMlFlowTagKeys.PIPELINE_STATUS}"
          ->
          s"${PipelineStatus.PIPELINE_FAILED} with message: $message (Search for pipeline ID $pipelineId in the cluster logs to find more details)"
        ))
  }

}
