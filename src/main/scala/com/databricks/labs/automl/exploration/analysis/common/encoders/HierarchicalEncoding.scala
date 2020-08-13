package com.databricks.labs.automl.exploration.analysis.common.encoders

import com.databricks.labs.automl.exploration.analysis.common.structures.{
  JSHierarchy,
  NodeData,
  PipelineNodeData,
  StringIndexerMappings
}

import com.databricks.labs.automl.exploration.analysis.common.structures.PayloadType._

import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.writePretty
import org.json4s.{Formats, FullTypeHints}

private[analysis] object HierarchicalEncoding {

  private def withElse[T](
    nodeInfo: NodeData
  )(conditionalCheck: => Boolean)(conditionalCode: => Option[T]): Option[T] = {
    if (conditionalCheck) conditionalCode else None
  }

  private def withElse[T](
    pipelineInfo: PipelineNodeData
  )(conditionalCheck: => Boolean)(conditionalCode: => Option[T]): Option[T] = {
    if (conditionalCheck) conditionalCode else None
  }

  private def convertTreeDataToJSFormat[T](
    data: T,
    fieldNames: Array[String],
    indexerMappings: Option[Array[StringIndexerMappings]]
  ): JSHierarchy = {

    val payloadType = data match {
      case _: PipelineNodeData => PIPELINE; case _: NodeData => MODEL
    }

    val fieldMappings =
      Encoders
        .getUpdatedFieldMappings(payloadType, fieldNames, indexerMappings)
        .map(x => x._2 -> x._1)
        .toMap

    payloadType match {
      case MODEL => {
        val nodeData = data.asInstanceOf[NodeData]

        JSHierarchy(
          name = nodeData.featureIndex match {
            case Some(x) => fieldMappings(x)
            case _       => "Leaf"
          },
          informationGain = withElse(nodeData)(
            nodeData.informationGain.isDefined
          )(Some(nodeData.informationGain.get)),
          continuousSplitThreshold = withElse(nodeData)(
            nodeData.continuousSplitThreshold.isDefined
          )(Some(nodeData.continuousSplitThreshold.get)),
          splitType = withElse(nodeData)(nodeData.splitType.isDefined)(
            Some(nodeData.splitType.get)
          ),
          prediction = nodeData.prediction,
          leftNodeCategories = withElse(nodeData)(
            nodeData.leftNodeCategories.isDefined
          )(Some(nodeData.leftNodeCategories.get.mkString(", "))),
          rightNodeCategories = withElse(nodeData)(
            nodeData.rightNodeCategories.isDefined
          )(Some(nodeData.rightNodeCategories.get.mkString(", "))),
          children = if (nodeData.treeNodeType == "node") {
            Some(
              Array(
                withElse(nodeData)(nodeData.leftChild.isDefined)(
                  Some(
                    convertTreeDataToJSFormat(
                      nodeData.leftChild.get,
                      fieldNames,
                      indexerMappings
                    )
                  )
                ),
                withElse(nodeData)(nodeData.rightChild.isDefined)(
                  Some(
                    convertTreeDataToJSFormat(
                      nodeData.rightChild.get,
                      fieldNames,
                      indexerMappings
                    )
                  )
                )
              )
            )
          } else None
        )
      }
      case PIPELINE => {
        val nodeData = data.asInstanceOf[PipelineNodeData]

        JSHierarchy(
          name = nodeData.featureIndex match {
            case Some(x) => fieldMappings(x)
            case _       => "Leaf"
          },
          informationGain = withElse(nodeData)(
            nodeData.informationGain.isDefined
          )(Some(nodeData.informationGain.get)),
          continuousSplitThreshold = withElse(nodeData)(
            nodeData.continuousSplitThreshold.isDefined
          )(Some(nodeData.continuousSplitThreshold.get)),
          splitType = withElse(nodeData)(nodeData.splitType.isDefined)(
            Some(nodeData.splitType.get)
          ),
          prediction = nodeData.prediction,
          leftNodeCategories = withElse(nodeData)(
            nodeData.leftNodeCategories.isDefined
          )(Some(nodeData.leftNodeCategories.get.mkString(", "))),
          rightNodeCategories = withElse(nodeData)(
            nodeData.rightNodeCategories.isDefined
          )(Some(nodeData.rightNodeCategories.get.mkString(", "))),
          children = if (nodeData.treeNodeType == "node") {
            Some(
              Array(
                withElse(nodeData)(nodeData.leftChild.isDefined)(
                  Some(
                    convertTreeDataToJSFormat(
                      nodeData.leftChild.get,
                      fieldNames,
                      indexerMappings
                    )
                  )
                ),
                withElse(nodeData)(nodeData.rightChild.isDefined)(
                  Some(
                    convertTreeDataToJSFormat(
                      nodeData.rightChild.get,
                      fieldNames,
                      indexerMappings
                    )
                  )
                )
              )
            )
          } else None
        )

      }
    }

  }

  private def convertJSCollectionToJSON(jsHierarchy: JSHierarchy): String = {

    implicit val jsonFormat: Formats =
      Serialization.formats(hints = FullTypeHints(List(JSHierarchy.getClass)))
    writePretty(jsHierarchy)

  }

  def performJSEncoding[T](
    data: T,
    fieldNames: Array[String],
    indexerMappings: Option[Array[StringIndexerMappings]]
  ): String = {

    convertJSCollectionToJSON(
      convertTreeDataToJSFormat(data, fieldNames, indexerMappings)
    )

  }

}
