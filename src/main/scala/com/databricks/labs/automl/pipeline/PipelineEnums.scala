package com.databricks.labs.automl.pipeline

object PipelineEnums extends Enumeration {

  type PipelineEnums = PipelineConstants

  val OHE_SUFFIX = PipelineConstants("_oh")
  val SI_SUFFIX = PipelineConstants("_si")
  val FEATURE_NAME_TEMP_SUFFIX = PipelineConstants("_r")

  case class PipelineConstants(value: String)
}