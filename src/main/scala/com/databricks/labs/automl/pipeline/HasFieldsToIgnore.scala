package com.databricks.labs.automl.pipeline

import org.apache.spark.ml.param.{Params, StringArrayParam}

/**
  * @author Jas Bali
  *
  */
trait HasFieldsToIgnore extends Params {

  final val fieldsToIgnore: StringArrayParam = new StringArrayParam(this, "fieldsToIgnore", "Columns To Ignore")

  def setFieldsToIgnore(value: Array[String]): this.type = set(fieldsToIgnore, value)

  def getFieldsToIgnore: Array[String] = $(fieldsToIgnore)
}
