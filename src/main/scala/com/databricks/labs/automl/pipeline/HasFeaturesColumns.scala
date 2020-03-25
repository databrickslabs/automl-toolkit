package com.databricks.labs.automl.pipeline

import org.apache.spark.ml.param.{Params, StringArrayParam}

/**
  * @author Jas Bali
  *
  */
trait HasFeaturesColumns extends Params {

  final val featureColumns: StringArrayParam = new StringArrayParam(this, "featureColumns", "List of feature column names")

  def setFeatureColumns(value: Array[String]): this.type = set(featureColumns, value)

  def getFeatureColumns: Array[String] = $(featureColumns)

}
