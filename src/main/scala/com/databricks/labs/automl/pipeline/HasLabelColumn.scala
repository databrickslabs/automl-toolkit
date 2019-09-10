package com.databricks.labs.automl.pipeline

import org.apache.spark.ml.param.{Param, Params}

/**
  * @author Jas Bali
  *
  */
trait HasLabelColumn extends Params{

  final val labelColumn: Param[String] = new Param[String](this, "labelColumn", "Label Column Name")

  def setLabelColumn(value: String): this.type = set(labelColumn, value)

  def getLabelColumn: String = $(labelColumn)
}
