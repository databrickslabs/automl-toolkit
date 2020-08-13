package com.databricks.labs.automl.exploration.analysis.trees

import com.databricks.labs.automl.exploration.analysis.common.encoders.ModelDebugStringParser
import com.databricks.labs.automl.exploration.analysis.common.structures.{
  AbstractVisualization,
  VisualizationOutput,
  VisualizationSettings
}
import com.databricks.labs.automl.exploration.analysis.trees.extractors.{
  ImportancesExtractor,
  VisualizationExtractor
}
import com.databricks.labs.automl.exploration.analysis.trees.scripts.HTMLGenerators
import org.apache.spark.ml.PipelineModel

class TreePipelineVisualization(pipeline: PipelineModel, mode: String)
    extends AbstractVisualization
    with VisualizationSettings {

  override def extractAllTreeDataAsString: String = {
    ModelDebugStringParser.getModelDecisionTextFromPipeline(pipeline)
  }

  override def extractAllTreeVisualization: Array[VisualizationOutput] = {
    checkMode(mode)
    VisualizationExtractor.extractModelVisualizationDataFromPipeline(
      pipeline,
      mode
    )
  }

  override def extractFirstTreeVisualization: String = {
    checkMode(mode)
    extractAllTreeVisualization.head.html

  }

  override def extractImportancesAsTable: String = {

    HTMLGenerators.buildFeatureImportancesTable(
      ImportancesExtractor
        .extractImportancesFromPipeline(pipeline)
    )

  }

  override def extractImportancesAsChart: String = {

    HTMLGenerators.buildFeatureImportanceChart(
      ImportancesExtractor
        .extractImportancesFromPipeline(pipeline)
    )

  }
}

object TreePipelineVisualization {

  def apply(pipeline: PipelineModel, mode: String): TreePipelineVisualization =
    new TreePipelineVisualization(pipeline, mode)

}
