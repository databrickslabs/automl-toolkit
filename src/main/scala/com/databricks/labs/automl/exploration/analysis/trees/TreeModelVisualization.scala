package com.databricks.labs.automl.exploration.analysis.trees

import com.databricks.labs.automl.exploration.analysis.common.encoders.ModelDebugStringParser
import com.databricks.labs.automl.exploration.analysis.common.structures.{
  AbstractVisualization,
  NoParam,
  ParamWrapper,
  VisualizationOutput,
  VisualizationSettings
}
import com.databricks.labs.automl.exploration.analysis.trees.extractors.{
  ImportancesExtractor,
  VisualizationExtractor
}
import com.databricks.labs.automl.exploration.analysis.trees.scripts.HTMLGenerators
import org.apache.spark.ml.feature.VectorAssembler

class TreeModelVisualization[T](
  model: T,
  mode: String,
  vectorAssembler: ParamWrapper[VectorAssembler] = NoParam,
  vectorInputCols: ParamWrapper[Array[String]] = NoParam
) extends AbstractVisualization
    with VisualizationSettings {

  override def extractAllTreeDataAsString: String = {

    ModelDebugStringParser.getModelDecisionText(
      model,
      vectorAssembler,
      vectorInputCols
    )

  }

  override def extractAllTreeVisualization: Array[VisualizationOutput] = {

    checkMode(mode)

    VisualizationExtractor.extractModelVisualizationDataFromModel(
      model,
      mode,
      vectorAssembler,
      vectorInputCols
    )

  }

  override def extractFirstTreeVisualization: String = {

    checkMode(mode)
    extractAllTreeVisualization.head.html

  }

  override def extractImportancesAsTable: String = {

    HTMLGenerators.buildFeatureImportancesTable(
      ImportancesExtractor
        .extractImportancesFromModel(model, vectorAssembler, vectorInputCols)
    )

  }

  override def extractImportancesAsChart: String = {

    HTMLGenerators.buildFeatureImportanceChart(
      ImportancesExtractor
        .extractImportancesFromModel(model, vectorAssembler, vectorInputCols)
    )

  }

}

object TreeModelVisualization {

  def apply[T](
    model: T,
    mode: String,
    vectorAssembler: ParamWrapper[VectorAssembler] = NoParam,
    vectorInputCols: ParamWrapper[Array[String]] = NoParam
  ): TreeModelVisualization[T] =
    new TreeModelVisualization(model, mode, vectorAssembler, vectorInputCols)

}
