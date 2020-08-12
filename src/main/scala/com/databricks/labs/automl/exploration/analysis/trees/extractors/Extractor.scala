package com.databricks.labs.automl.exploration.analysis.trees.extractors

import com.databricks.labs.automl.exploration.analysis.common.structures.{
  ExtractorType,
  NodeData,
  NodeDetermination,
  PipelineNodeData
}
import org.apache.spark.ml.tree.{
  CategoricalSplit,
  ContinuousSplit,
  InternalNode,
  Node
}

private[analysis] object Extractor {

  import com.databricks.labs.automl.exploration.analysis.common.structures.NodeType._
  import com.databricks.labs.automl.exploration.analysis.common.structures.PayloadType._
  import com.databricks.labs.automl.exploration.analysis.common.structures.SplitType._

  def extractRules(
    treeNode: Node,
    payloadType: PayloadType,
    indexerMappings: Option[Map[Int, Map[Int, String]]]
  ): ExtractorType = {

    val nodeType = NodeDetermination.nodeType(treeNode)
    val internalNodeData = nodeType match {
      case NODE => Some(treeNode.asInstanceOf[InternalNode])
      case LEAF => None
    }

    val splitType = nodeType match {
      case NODE => Some(NodeDetermination.splitType(internalNodeData.get.split))
      case _    => None
    }

    val featureIndex = nodeType match {
      case NODE => Some(internalNodeData.get.split.featureIndex)
      case _    => None
    }
    val informationGain = nodeType match {
      case NODE => Some(internalNodeData.get.gain)
      case _    => None
    }
    val continuousSplitThreshold = splitType.getOrElse(None) match {
      case CONTINUOUS =>
        Some(internalNodeData.get.split.asInstanceOf[ContinuousSplit].threshold)
      case _ => None
    }
    val treeNodeType = nodeType match {
      case NODE => "node"
      case _    => "leaf"
    }
    val nodeSplitType = nodeType match {
      case NODE =>
        splitType.get match {
          case CONTINUOUS  => Some("continuous")
          case CATEGORICAL => Some("categorical")
          case _           => None
        }
      case _ => None
    }

    val leftNodeCategories = payloadType match {
      case MODEL =>
        splitType.getOrElse(None) match {
          case CATEGORICAL =>
            Some(
              internalNodeData.get.split
                .asInstanceOf[CategoricalSplit]
                .leftCategories
            )
          case _ => None
        }
      case _ => None
    }

    val rightNodeCategories = payloadType match {
      case MODEL =>
        splitType.getOrElse(None) match {
          case CATEGORICAL =>
            Some(
              internalNodeData.get.split
                .asInstanceOf[CategoricalSplit]
                .rightCategories
            )
          case _ => None
        }
      case _ => None
    }

    val pipelineLeftNodeCategories = payloadType match {
      case PIPELINE =>
        splitType.getOrElse(None) match {
          case CATEGORICAL =>
            Some(
              internalNodeData.get.split
                .asInstanceOf[CategoricalSplit]
                .leftCategories
                .map(
                  x =>
                    indexerMappings
                      .get(internalNodeData.get.split.featureIndex)(x.toInt)
                )
            )

          case _ => None
        }
      case _ => None
    }
    val pipelineRightNodeCategories = payloadType match {
      case PIPELINE =>
        splitType.getOrElse(None) match {
          case CATEGORICAL =>
            Some(
              internalNodeData.get.split
                .asInstanceOf[CategoricalSplit]
                .rightCategories
                .map(
                  x =>
                    indexerMappings
                      .get(internalNodeData.get.split.featureIndex)(x.toInt)
                )
            )
          case _ => None
        }
      case _ => None
    }
    val leftChild = nodeType match {
      case NODE =>
        Some(
          extractRules(
            internalNodeData.get.leftChild,
            payloadType,
            indexerMappings
          )
        )
      case _ => None
    }
    val rightChild = nodeType match {
      case NODE =>
        Some(
          extractRules(
            internalNodeData.get.rightChild,
            payloadType,
            indexerMappings
          )
        )
      case _ => None
    }
    val prediction = treeNode.prediction

    payloadType match {
      case PIPELINE =>
        PipelineNodeData(
          featureIndex = featureIndex,
          informationGain = informationGain,
          continuousSplitThreshold = continuousSplitThreshold,
          treeNodeType = treeNodeType,
          splitType = nodeSplitType,
          leftNodeCategories = pipelineLeftNodeCategories,
          rightNodeCategories = pipelineRightNodeCategories,
          leftChild = leftChild.asInstanceOf[Option[PipelineNodeData]],
          rightChild = rightChild.asInstanceOf[Option[PipelineNodeData]],
          prediction = prediction
        )
      case MODEL =>
        NodeData(
          featureIndex = featureIndex,
          informationGain = informationGain,
          continuousSplitThreshold = continuousSplitThreshold,
          treeNodeType = treeNodeType,
          splitType = nodeSplitType,
          leftNodeCategories = leftNodeCategories,
          rightNodeCategories = rightNodeCategories,
          leftChild = leftChild.asInstanceOf[Option[NodeData]],
          rightChild = rightChild.asInstanceOf[Option[NodeData]],
          prediction = prediction
        )
    }

  }

}
