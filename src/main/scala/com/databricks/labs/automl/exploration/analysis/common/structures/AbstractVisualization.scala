package com.databricks.labs.automl.exploration.analysis.common.structures

private[analysis] abstract class AbstractVisualization {

  def extractAllTreeDataAsString: String
  def extractAllTreeVisualization: Array[VisualizationOutput]
  def extractFirstTreeVisualization: String
  def extractImportancesAsTable: String
  def extractImportancesAsChart: String

}
