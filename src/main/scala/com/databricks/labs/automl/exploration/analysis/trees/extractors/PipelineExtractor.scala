package com.databricks.labs.automl.exploration.analysis.trees.extractors

import com.databricks.labs.automl.exploration.analysis.common.AnalysisUtilities
import com.databricks.labs.automl.exploration.analysis.common.encoders.HierarchicalEncoding
import com.databricks.labs.automl.exploration.analysis.common.structures.{
  PayloadDetermination,
  PipelineNodeData,
  PipelineReport
}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{
  DecisionTreeClassificationModel,
  GBTClassificationModel,
  RandomForestClassificationModel
}
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.regression.{
  DecisionTreeRegressionModel,
  GBTRegressionModel,
  RandomForestRegressionModel
}
import org.apache.spark.ml.tree.Node

private[analysis] class PipelineExtractor(pipeline: PipelineModel) {

  private def getStringIndexerLabels(
    pipeline: PipelineModel
  ): Map[String, Map[Int, String]] = {

    pipeline.stages
      .collect {
        case x: StringIndexerModel =>
          val indexer = x.asInstanceOf[StringIndexerModel]
          val labels = indexer.getHandleInvalid match {
            case "keep" => indexer.labels :+ "Unknown"
            case _      => indexer.labels
          }
          Map(indexer.getOutputCol -> labels.indices.zip(labels).toMap)
        case x: PipelineModel => getStringIndexerLabels(x)
      }
      .flatten
      .toMap

  }

  private def resolveIndexerMappings: Map[Int, Map[Int, String]] = {

    val stringIndexers = getStringIndexerLabels(pipeline)
    val finalFeatures = AnalysisUtilities.getFinalFeaturesFromPipeline(pipeline)

    stringIndexers.keys.toArray
      .map(x => finalFeatures(x) -> stringIndexers(x))
      .toMap

  }

  def extractRootNode: Array[Node] = {
    AnalysisUtilities.getModelFromPipeline(pipeline).last match {
      case x: RandomForestClassificationModel =>
        new TreeExtractor(x.asInstanceOf[RandomForestClassificationModel]).extractRootNode
      case x: RandomForestRegressionModel =>
        new TreeExtractor(x.asInstanceOf[RandomForestRegressionModel]).extractRootNode
      case x: DecisionTreeClassificationModel =>
        new TreeExtractor(x.asInstanceOf[DecisionTreeClassificationModel]).extractRootNode
      case x: DecisionTreeRegressionModel =>
        new TreeExtractor(x.asInstanceOf[DecisionTreeRegressionModel]).extractRootNode
      case x: GBTClassificationModel =>
        new TreeExtractor(x.asInstanceOf[GBTClassificationModel]).extractRootNode
      case x: GBTRegressionModel =>
        new TreeExtractor(x.asInstanceOf[GBTRegressionModel]).extractRootNode
    }
  }

  def extractPipelineInformation: Array[PipelineReport] = {

    val payloadType = PayloadDetermination.payloadType(pipeline)

    val indexMappings = resolveIndexerMappings

    extractRootNode
      .map(n => Extractor.extractRules(n, payloadType, Some(indexMappings)))
      .zipWithIndex
      .map(x => PipelineReport(x._2, x._1.asInstanceOf[PipelineNodeData]))

  }

  def getVisualizationData: Array[String] = {

    val fieldNames = AnalysisUtilities.getPipelineVectorFields(pipeline)
    val pipelineExtract = extractPipelineInformation
    val indexData = AnalysisUtilities.getStringIndexerMapping(pipeline)

    pipelineExtract.map(
      x =>
        HierarchicalEncoding
          .performJSEncoding(x.data, fieldNames, indexData)
    )
  }

}
