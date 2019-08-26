package com.databricks.labs.automl.pipeline

import org.apache.spark.ml.param.{BooleanParam, Params}

trait HasTransformCalculated extends Params {

  final val transformCalculated: BooleanParam = new BooleanParam(this, "varianceCalculated", "Flag to help for predict pipeline to avoid calculating variance again")

  def setTransformCalculated(value: Boolean): this.type = set(transformCalculated, value)

  def getTransformCalculated: Boolean = $(transformCalculated)

}
