package com.databricks.labs.automl.exploration.analysis.common.encoders

import com.databricks.labs.automl.exploration.analysis.common.structures.{
  FeatureImportanceData,
  ImportanceMapping
}
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.writePretty
import org.json4s.{Formats, FullTypeHints}

private[analysis] object Converters {

  def mapFieldsToImportances(
    importanceData: FeatureImportanceData
  ): Array[String] = {

    Encoders
      .getUpdatedFieldMappings(
        importanceData.payloadType,
        importanceData.featureVectorNames,
        importanceData.indexerMappings
      )
      .sortWith(_._2 < _._2)
      .map(_._1)
      .zip(importanceData.importances.toArray)
      .sortWith(_._2 > _._2)
      .map(x => Map("feature" -> x._1, "importance" -> x._2))
      .map(x => s"<tr><td>${x("feature")}</td><td>${x("importance")}</td></tr>")
  }

  def extractFieldImportancesAsJSON(
    importanceData: FeatureImportanceData
  ): String = {

    val mapping = Encoders
      .getUpdatedFieldMappings(
        importanceData.payloadType,
        importanceData.featureVectorNames,
        importanceData.indexerMappings
      )
      .sortWith(_._2 < _._2)
      .map(_._1)
      .zip(importanceData.importances.toArray)
      .sortWith(_._2 > _._2)
      .map(x => ImportanceMapping(x._1, x._2))

    convertFItoJSON(mapping)
  }

  private def convertFItoJSON(
    importanceData: Array[ImportanceMapping]
  ): String = {

    implicit val jsonFormat: Formats =
      Serialization.formats(
        hints = FullTypeHints(List(ImportanceMapping.getClass))
      )
    writePretty(importanceData)

  }

}
