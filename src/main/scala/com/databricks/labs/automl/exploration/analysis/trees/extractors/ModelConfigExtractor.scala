package com.databricks.labs.automl.exploration.analysis.trees.extractors

import com.databricks.labs.automl.exploration.analysis.common.structures.ModelData
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

private[analysis] object ModelConfigExtractor {

  def extractModelData[T](model: T): ModelData = {

    model match {

      case x: DecisionTreeRegressionModel =>
        ModelData(x.depth, x.numFeatures, x.numNodes)
      case x: DecisionTreeClassificationModel =>
        ModelData(x.depth, x.numFeatures, x.numNodes)
      case x: RandomForestRegressionModel =>
        ModelData(x.getMaxDepth, x.numFeatures, x.totalNumNodes / x.getNumTrees)
      case x: RandomForestClassificationModel =>
        ModelData(x.getMaxDepth, x.numFeatures, x.totalNumNodes / x.getNumTrees)
      case x: GBTRegressionModel =>
        ModelData(x.getMaxDepth, x.numFeatures, x.totalNumNodes / x.getNumTrees)
      case x: GBTClassificationModel =>
        ModelData(x.getMaxDepth, x.numFeatures, x.totalNumNodes / x.getNumTrees)
    }

  }

}
