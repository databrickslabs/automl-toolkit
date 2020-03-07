package com.databricks.labs.automl.pipeline

import org.apache.spark.ml.param.{Params, StringArrayParam}

/**
  * @author Jas Bali
  *
  */
trait HasFieldsRemoved extends Params {

  final val fieldsRemoved: StringArrayParam = new StringArrayParam(this, "fieldsRemoved", "fieldsRemoved")

  def setFieldsRemoved(value: Array[String]): this.type = set(fieldsRemoved, value)

  def getFieldsRemoved: Array[String] = $(fieldsRemoved)
}
