package com.databricks.labs.automl.utils

import com.databricks.labs.automl.params.MainConfig

/**
  * @author Jas Bali
  * @since 0.6.1
  * Enums for reporting pipeline to MLflow
  */
object PipelineMlFlowTagKeys {

  lazy final val PIPELINE_MODEL_SAVE_PATH_KEY = "BestPipelineModelSavePath"
  lazy final val PIPELINE_TRAIN_DF_PATH_KEY = "FeatureEngineeredTrainDfPath"
  lazy final val PIPELINE_STATUS = "PipelineExecutionCurrentStatus"
  lazy final val PIPELINE_ID = "PipelineExecutionCurrentStatus"
}
object PipelineStatus extends Enumeration {

  type PipelineStatus = PipelineStatusEnum

  val PIPELINE_STARTED = PipelineStatusEnum("STARTED")
  val PIPELINE_RUNNING = PipelineStatusEnum("RUNNING")
  val PIPELINE_COMPLETED = PipelineStatusEnum("COMPLETED")
  val PIPELINE_FAILED = PipelineStatusEnum("FAILED")

  case class PipelineStatusEnum(key: String) extends Val
}