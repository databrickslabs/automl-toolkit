package com.databricks.labs.automl.pipeline

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

  val PIPELINE_LABEL_REFACTOR_NEEDED_KEY = PipelineVarsPair("labelRefactorNeeded", Boolean.getClass)

  case class PipelineVarsPair(key: String, keyType: Class[_]) extends Val
}