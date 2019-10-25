package com.databricks.labs.automl.pipeline.inference

import com.databricks.labs.automl.executor.config.LoggingConfig
import com.databricks.labs.automl.params.MLFlowConfig
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.spark.ml.PipelineModel

/**
  * @author Jas Bali
  * @since 0.6.1
  * Utility functions for running inference directly against an MlFlow Run ID
  */
object PipelineModelInference {

  /**
    * Run Inference directly against a given MlFlow Run ID
    * @param runId
    * @param mlFlowConfig
    * @return
    */
  def getPipelineModelByMlFlowRunId(runId: String, mlFlowConfig: MLFlowConfig): PipelineModel = {
    PipelineModel.load(AutoMlPipelineMlFlowUtils.getPipelinePathByRunId(runId, mlFlowConfig))
  }

  /**
    * Overidden method for running inference using LoggingConfig
    *
    * @param runId
    * @param loggingConfig
    * @return
    */
  def getPipelineModelByMlFlowRunId(runId: String, loggingConfig: LoggingConfig): PipelineModel = {
    getPipelineModelByMlFlowRunId(
      runId,
      MLFlowConfig(
        loggingConfig.mlFlowTrackingURI,
        loggingConfig.mlFlowExperimentName,
        loggingConfig.mlFlowAPIToken,
        loggingConfig.mlFlowModelSaveDirectory,
        loggingConfig.mlFlowLoggingMode,
        loggingConfig.mlFlowBestSuffix,
        loggingConfig.mlFlowCustomRunTags
      ))
  }

}
