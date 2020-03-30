package com.databricks.labs.automl.exploration.structures

import org.apache.spark.sql.DataFrame

object CutoffTypes extends Enumeration {
  type CutoffTypes = Value
  val None, Threshold, Count = Value
}

object FeatureImportanceModelFamily extends Enumeration {
  type FeatureImportanceModelFamily = Value
  val RandomForest, XGBoost = Value
}

object ModelType extends Enumeration {
  type ModelType = Value
  val Regressor, Classifier = Value
}

sealed trait FIReturn {
  def data: DataFrame
  def fieldsInVector: Array[String]
  def allFields: Array[String]
}

abstract case class FeatureImportanceOutput() extends FIReturn
abstract case class FeatureImportanceReturn(importances: DataFrame,
                                            topFields: Array[String])
    extends FIReturn
