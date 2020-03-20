package com.databricks.labs.automl.exploration.tools

import org.apache.spark.sql.DataFrame

class PCAReducer(data: DataFrame) {

  //TOOD in progress AML-

  final private val K_VALUE = 2

  var labelColumn = "label"

  def setLabelColumn(value: String): this.type = {
    labelColumn = value
    this
  }

  def getLabelColumn: String = labelColumn

  // create feature vector

}
