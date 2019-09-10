package com.databricks.labs.automl.pipeline

import org.apache.spark.ml.param.{BooleanParam, Params}

/**
  * @author Jas Bali
  *
  */
trait HasTransformCalculated extends Params {

  final val transformCalculated: BooleanParam = new BooleanParam(this, "transformCalculated", "Flag to help for predict pipeline to avoid calculating estimators again")

  def setTransformCalculated(value: Boolean): this.type = set(transformCalculated, value)

  def getTransformCalculated: Boolean = $(transformCalculated)
}
