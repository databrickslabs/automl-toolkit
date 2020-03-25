package com.databricks.labs.automl.pipeline

import org.apache.spark.ml.param.{Param, Params}

/**
  * @author Jas Bali
  * @since 0.6.1
  * trait for decorating all pipeline stages with pipeline ID.
  * Helpful when troubleshooting logs with a given pipeline ID (eg, fetched from MLflow)
  */
trait HasPipelineId extends Params {

  final val pipelineId: Param[String] = new Param[String](this, "pipelineId", "UUID for AutoML pipeline")

  def setPipelineId(value: String): this.type = set(pipelineId, value)

  def getPipelineId: String = $(pipelineId)
}
