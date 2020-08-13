package com.databricks.labs.automl.exploration.analysis.trees.extractors

import com.databricks.labs.automl.exploration.analysis.common.encoders.HierarchicalEncoding
import com.databricks.labs.automl.exploration.analysis.common.structures.{
  NodeData,
  PayloadDetermination,
  TreeModelExtractor,
  TreesReport
}
import org.apache.spark.ml.classification.{
  DecisionTreeClassificationModel,
  GBTClassificationModel,
  RandomForestClassificationModel
}
import org.apache.spark.ml.regression.{
  DecisionTreeRegressionModel,
  GBTRegressionModel,
  RandomForestRegressionModel
}
import org.apache.spark.ml.tree.Node

private[analysis] class TreeExtractor[T: TreeModelExtractor](model: T) {

  def extractRootNode: Array[Node] = {

    model match {
      case _: DecisionTreeClassificationModel =>
        Array(model.asInstanceOf[DecisionTreeClassificationModel].rootNode)
      case _: DecisionTreeRegressionModel =>
        Array(model.asInstanceOf[DecisionTreeRegressionModel].rootNode)
      case _: RandomForestClassificationModel =>
        model
          .asInstanceOf[RandomForestClassificationModel]
          .trees
          .map(_.rootNode)
      case _: RandomForestRegressionModel =>
        model.asInstanceOf[RandomForestRegressionModel].trees.map(_.rootNode)
      case _: GBTClassificationModel =>
        model.asInstanceOf[GBTClassificationModel].trees.map(_.rootNode)
      case _: GBTRegressionModel =>
        model.asInstanceOf[GBTRegressionModel].trees.map(_.rootNode)
    }

  }

  def extractTreeInformation: Array[TreesReport] = {

    val payloadType = PayloadDetermination.payloadType(model)

    extractRootNode
      .map(n => Extractor.extractRules(n, payloadType, None))
      .zipWithIndex
      .map(x => TreesReport(x._2, x._1.asInstanceOf[NodeData]))

  }

  def getVisualizationData(vectorFieldNames: Array[String]): Array[String] = {

    extractTreeInformation.map(
      x =>
        HierarchicalEncoding.performJSEncoding(x.data, vectorFieldNames, None)
    )

  }

}
