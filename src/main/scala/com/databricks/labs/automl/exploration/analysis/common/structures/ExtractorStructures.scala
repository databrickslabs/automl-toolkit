package com.databricks.labs.automl.exploration.analysis.common.structures

import com.databricks.labs.automl.exploration.analysis.common.structures.PayloadType.PayloadType

private[analysis] sealed trait ExtractorType

private[analysis] case class NodeData(
  featureIndex: Option[Int],
  informationGain: Option[Double],
  continuousSplitThreshold: Option[Double],
  treeNodeType: String,
  splitType: Option[String],
  leftNodeCategories: Option[Array[Double]],
  rightNodeCategories: Option[Array[Double]],
  leftChild: Option[NodeData],
  rightChild: Option[NodeData],
  prediction: Double
) extends ExtractorType

private[analysis] case class PipelineNodeData(
  featureIndex: Option[Int],
  informationGain: Option[Double],
  continuousSplitThreshold: Option[Double],
  treeNodeType: String,
  splitType: Option[String],
  leftNodeCategories: Option[Array[String]],
  rightNodeCategories: Option[Array[String]],
  leftChild: Option[PipelineNodeData],
  rightChild: Option[PipelineNodeData],
  prediction: Double
) extends ExtractorType

private[analysis] case class FeatureIndexRenamingStructure(
  featureName: String,
  replacementText: String
)

private[analysis] case class PipelineReport(tree: Int, data: PipelineNodeData)

private[analysis] case class TreesReport(tree: Int, data: NodeData)

private[analysis] case class ForestReport(featureImportanceHTML: String,
                                          totalNodesInForest: Int,
                                          classCount: Option[Int],
                                          treeWeights: Map[String, Double],
                                          featureCount: Int)

private[analysis] case class JSHierarchy(
  name: String,
  informationGain: Option[Double],
  continuousSplitThreshold: Option[Double],
  splitType: Option[String],
  prediction: Double,
  leftNodeCategories: Option[String],
  rightNodeCategories: Option[String],
  children: Option[Array[Option[JSHierarchy]]]
)

private[analysis] case class StringIndexerMappings(before: String,
                                                   after: String)

private[analysis] case class VisualizationOutput(tree: Int, html: String)

private[analysis] case class FeatureImportanceData(
  importances: org.apache.spark.ml.linalg.Vector,
  featureVectorNames: Array[String],
  payloadType: PayloadType,
  indexerMappings: Option[Array[StringIndexerMappings]]
)

private[analysis] case class ImportanceMapping(feature: String,
                                               importance: Double)

private[analysis] case class ModelData(modelDepth: Int,
                                       featureCount: Int,
                                       nodeCount: Int)
