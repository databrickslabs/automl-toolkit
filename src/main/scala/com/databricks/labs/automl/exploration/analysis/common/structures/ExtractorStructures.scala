package com.databricks.labs.automl.exploration.analysis.common.structures

import com.databricks.labs.automl.exploration.analysis.common.structures.PayloadType.PayloadType

sealed trait ExtractorType

case class NodeData(featureIndex: Option[Int],
                    informationGain: Option[Double],
                    continuousSplitThreshold: Option[Double],
                    treeNodeType: String,
                    splitType: Option[String],
                    leftNodeCategories: Option[Array[Double]],
                    rightNodeCategories: Option[Array[Double]],
                    leftChild: Option[NodeData],
                    rightChild: Option[NodeData],
                    prediction: Double)
    extends ExtractorType

case class PipelineNodeData(featureIndex: Option[Int],
                            informationGain: Option[Double],
                            continuousSplitThreshold: Option[Double],
                            treeNodeType: String,
                            splitType: Option[String],
                            leftNodeCategories: Option[Array[String]],
                            rightNodeCategories: Option[Array[String]],
                            leftChild: Option[PipelineNodeData],
                            rightChild: Option[PipelineNodeData],
                            prediction: Double)
    extends ExtractorType

case class FeatureIndexRenamingStructure(featureName: String,
                                         replacementText: String)

case class PipelineReport(tree: Int, data: PipelineNodeData)

case class TreesReport(tree: Int, data: NodeData)

case class ForestReport(featureImportanceHTML: String,
                        totalNodesInForest: Int,
                        classCount: Option[Int],
                        treeWeights: Map[String, Double],
                        featureCount: Int)

case class JSHierarchy(name: String,
                       informationGain: Option[Double],
                       continuousSplitThreshold: Option[Double],
                       splitType: Option[String],
                       prediction: Double,
                       leftNodeCategories: Option[String],
                       rightNodeCategories: Option[String],
                       children: Option[Array[Option[JSHierarchy]]])

case class StringIndexerMappings(before: String, after: String)

case class VisualizationOutput(tree: Int, html: String)

case class FeatureImportanceData(
  importances: org.apache.spark.ml.linalg.Vector,
  featureVectorNames: Array[String],
  payloadType: PayloadType,
  indexerMappings: Option[Array[StringIndexerMappings]]
)

case class ImportanceMapping(feature: String, importance: Double)

case class ModelData(modelDepth: Int, featureCount: Int, nodeCount: Int)
