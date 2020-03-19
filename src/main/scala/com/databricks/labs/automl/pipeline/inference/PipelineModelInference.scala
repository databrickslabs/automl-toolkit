package com.databricks.labs.automl.pipeline.inference

import com.databricks.labs.automl.executor.config.LoggingConfig
import com.databricks.labs.automl.params.{MLFlowConfig, MainConfig}
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
    * @param mainConfig
    * @return
    */
  def getPipelineModelByMlFlowRunId(runId: String, mainConfig: MainConfig): PipelineModel = {
    PipelineModel.load(AutoMlPipelineMlFlowUtils.getPipelinePathByRunId(runId, mainConfig))
  }

}
