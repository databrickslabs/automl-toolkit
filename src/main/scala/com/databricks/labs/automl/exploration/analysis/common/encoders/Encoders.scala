package com.databricks.labs.automl.exploration.analysis.common.encoders

import com.databricks.labs.automl.exploration.analysis.common.structures.{
  FeatureIndexRenamingStructure,
  NodeData,
  PipelineNodeData,
  StringIndexerMappings
}
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.writePretty
import org.json4s.{Formats, FullTypeHints}

private[analysis] object Encoders {

  import com.databricks.labs.automl.exploration.analysis.common.structures.PayloadType._

  private def encodeAsJSON[T](data: T, payloadType: PayloadType): String = {

    implicit val jsonFormat: Formats =
      Serialization.formats(hints = payloadType match {
        case PIPELINE => FullTypeHints(List(PipelineNodeData.getClass))
        case MODEL    => FullTypeHints(List(NodeData.getClass))
      })
    payloadType match {
      case PIPELINE => writePretty(data.asInstanceOf[PipelineNodeData])
      case MODEL    => writePretty(data.asInstanceOf[NodeData])
    }

  }

  def getUpdatedFieldMappings(
    payloadType: PayloadType,
    featureVectorNames: Array[String],
    indexerMappings: Option[Array[StringIndexerMappings]]
  ): Array[(String, Int)] = {

    payloadType match {
      case MODEL => featureVectorNames.zipWithIndex
      case PIPELINE =>
        featureVectorNames.zipWithIndex.map(x => {
          val mappings = indexerMappings.get
          val outputMappings = mappings.map(_.after)
          x._1 match {
            case y if outputMappings.contains(y) =>
              (mappings.filter(_.after == x._1).head.before, x._2)
            case _ => (x._1, x._2)
          }
        })
    }
  }

  def convertTreesDataToJSON[T](
    data: Array[T],
    payloadType: PayloadType,
    featureVectorNames: Array[String],
    indexerMappings: Option[Array[StringIndexerMappings]]
  ): String = {

    val extractedJSONData = encodeAsJSON(data, payloadType)

    getUpdatedFieldMappings(payloadType, featureVectorNames, indexerMappings)
      .map(
        x =>
          FeatureIndexRenamingStructure(x._1, s""""featureIndex" : ${x._2}""")
      )
      .foldLeft(extractedJSONData) {
        case (treeText, field) =>
          treeText.replaceAll(
            field.replacementText,
            s""""featureIndex" : "${field.featureName}""""
          )
      }
  }

}
