package com.databricks.labs.automl.pipeline

import java.util.UUID

import org.apache.spark.ml.{PipelineModel, Transformer}

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.mleap.SparkUtil

object PipelineInternalUtils {

  def mergePipelineModels(pipelineModels: Array[PipelineModel]): PipelineModel = {
    mergePipelineModelsInternal(pipelineModels)
  }

  private def mergePipelineModelsInternal(pipelineModels: Array[PipelineModel]): PipelineModel = {
    SparkUtil.createPipelineModel(
      "final_ml_pipeline_" + UUID.randomUUID().toString,
      pipelineModels.flatMap(item => item.stages)
    )
  }

  def addTransformersToPipelineModels(pipelineModel: PipelineModel, transformers: Array[_<: Transformer]): PipelineModel = {
    SparkUtil.createPipelineModel(
      "final_ml_pipeline_" + UUID.randomUUID().toString,
      pipelineModel.stages ++ transformers)
  }

  def createPipelineModelFromStages(stages: Array[Transformer]): PipelineModel = {
    SparkUtil.createPipelineModel(
      "final_ml_pipeline_" + UUID.randomUUID().toString,
      stages
    )
  }
}
