package com.databricks.labs.automl.pipeline

import org.apache.spark.ml.param.{Param, Params}

/**
  * @author Jas Bali
  *
  */
trait HasFeatureColumn extends Params {

  final val featureCol: Param[String] = new Param[String](this, "featureCol", "Feature Column Name")

  def setFeatureCol(value: String): this.type = set(featureCol, value)

  def getFeatureCol: String = $(featureCol)
}
