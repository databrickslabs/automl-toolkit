package com.databricks.labs.automl.pipeline

import org.apache.spark.ml.param.{Params, StringArrayParam}

/**
  * Trait for defining whether interaction columns have been set for the application of Feature Interactions
  * @since 0.6.2
  * @author Ben Wilson, Databricks
  */
trait HasInteractionColumns extends Params {

  final val leftColumns: StringArrayParam = new StringArrayParam(
    this,
    "leftColumns",
    "Left side columns for interaction"
  )
  final val rightColumns: StringArrayParam = new StringArrayParam(
    this,
    "rightColumns",
    "Right side columns for interaction"
  )

  def setLeftColumns(value: Array[String]): this.type = set(leftColumns, value)
  def setRightColumns(value: Array[String]): this.type =
    set(rightColumns, value)

  def getLeftColumns: Array[String] = $(leftColumns)
  def getRightColumns: Array[String] = $(rightColumns)

  def getInteractionColumns: Array[(String, String)] =
    ($(leftColumns) zip $(rightColumns))

}
