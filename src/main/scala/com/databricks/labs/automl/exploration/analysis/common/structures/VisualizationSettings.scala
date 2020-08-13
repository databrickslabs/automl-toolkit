package com.databricks.labs.automl.exploration.analysis.common.structures

private[analysis] trait VisualizationSettings {

  final val ALLOWABLE_MODES: Array[String] =
    Array("static", "dynamic", "lightweight")

  def checkMode(mode: String): Unit = {
    require(
      ALLOWABLE_MODES.contains(mode),
      s"The mode supplied '${mode}' is not supported.  Must be one of: ${ALLOWABLE_MODES.mkString(", ")}"
    )
  }

}
