package com.databricks.labs.automl.pipeline.inference

import com.databricks.labs.automl.executor.config.LoggingConfig
import com.databricks.labs.automl.params.{MLFlowConfig, MainConfig}
import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, InitDbUtils}
import org.apache.spark.ml.PipelineModel

/**
  * @author Jas Bali
  * @since 0.6.1
  * Utility functions for running inference directly against an MlFlow Run ID
  */
object PipelineModelInference {


  /**
    *
    * @param runId String of MLFlow runId to be used for Inference
    * @param loggingConfig Deprecated -- logging config for older pipelines
    * @return
    */
  @deprecated("Only for legacy pipelines without main config tracked by MLFlow. Use " +
    "signature (runId: String, mainConfig: mainConfig: MainConfig) or " +
    "(runId: String)", "0.7.1")
  def getPipelineModelByMlFlowRunId(runId: String, loggingConfig: LoggingConfig): PipelineModel = {
    PipelineModel.load(AutoMlPipelineMlFlowUtils.getPipelinePathByRunId(runId, loggingConfig=Some(loggingConfig)))
  }

  /***
    * String of MLFlow runId to be used for Inference
    * @param runId
    * @param mainConfig
    * @return
    */
  def getPipelineModelByMlFlowRunId(runId: String, mainConfig: MainConfig): PipelineModel = {
    PipelineModel.load(AutoMlPipelineMlFlowUtils.getPipelinePathByRunId(runId, mainConfig=Some(mainConfig)))
  }

  /**
    * String of MLFlow runId to be used for Inference
    * @param runId
    * @return
    */
  def getPipelineModelByMlFlowRunId(runId: String): PipelineModel = {
    PipelineModel.load(AutoMlPipelineMlFlowUtils.getPipelinePathByRunId(runId, None))
  }

}
