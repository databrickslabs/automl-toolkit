package com.databricks.labs.automl.exploration.analysis.common.structures

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

private[analysis] class TreeModelExtractor[T]
private[analysis] object TreeModelExtractor {
  implicit object DecisionTreeRegressorExtractor
      extends TreeModelExtractor[DecisionTreeRegressionModel]
  implicit object DecisionTreeClassifierExtractor
      extends TreeModelExtractor[DecisionTreeClassificationModel]
  implicit object RandomForestRegressorExtractor
      extends TreeModelExtractor[RandomForestRegressionModel]
  implicit object RandomForestClassificationExtractor
      extends TreeModelExtractor[RandomForestClassificationModel]
  implicit object GBTRegressorExtractor
      extends TreeModelExtractor[GBTRegressionModel]
  implicit object GBTClassifierExtractor
      extends TreeModelExtractor[GBTClassificationModel]
}
