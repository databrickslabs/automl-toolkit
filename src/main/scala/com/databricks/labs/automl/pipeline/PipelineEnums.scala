package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.params.{MLFlowConfig, MainConfig}

object PipelineEnums extends Enumeration {

  type PipelineEnums = PipelineConstants

  val OHE_SUFFIX = PipelineConstants("_oh")
  val SI_SUFFIX = PipelineConstants("_si")
  val FEATURE_NAME_TEMP_SUFFIX = PipelineConstants("_r")

  val LABEL_STRING_INDEXER_STAGE_NAME = PipelineConstants("LabelStringIndexer")

  case class PipelineConstants(value: String) extends Val
}


object PipelineVars extends Enumeration {

  type PipelineVars = PipelineVarsPair

  val PIPELINE_LABEL_REFACTOR_NEEDED_KEY = PipelineVarsPair("labelRefactorNeeded", classOf[Boolean])
  val MLFLOW_RUN_ID = PipelineVarsPair("MlFlowRunId", classOf[String])
  val MAIN_CONFIG = PipelineVarsPair("MainConfig", classOf[MainConfig])
  val PIPELINE_STATUS = PipelineVarsPair("PipelineStatus", classOf[String])
  val KSAMPLER_STAGES = PipelineVarsPair("KSamplerStages", classOf[String])

  case class PipelineVarsPair(key: String, keyType: Class[_]) extends Val
}