package com.databricks.labs.automl.pipeline

import java.util.UUID
import org.apache.spark.ml.PipelineModel
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.mleap.SparkUtil

object PipelineInternalUtils {

   def mergePipelineModels(pipelineModels: ArrayBuffer[PipelineModel]): PipelineModel = {
    SparkUtil.createPipelineModel(
      "final_ml_pipeline_" + UUID.randomUUID().toString,
      pipelineModels.flatMap(item => item.stages).toArray
    )
  }

}
