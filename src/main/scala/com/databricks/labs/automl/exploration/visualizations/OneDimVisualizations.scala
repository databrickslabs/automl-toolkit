package com.databricks.labs.automl.exploration.visualizations

import com.databricks.labs.automl.exploration.tools.OneDimStatsData
import vegas.DSL.ExtendedUnitSpecBuilder
import vegas._

object OneDimVisualizations {

  def renderHTML(plot: ExtendedUnitSpecBuilder, name: String): String = {
    plot.html.pageHTML(name)
  }

  def generateOneDimPlots(results: OneDimStatsData) = {

    val plot = Vegas("Variance")
      .withData(Seq(Map("field" -> "A", "variance" -> results.variance)))
      .encodeX("field", Nom)
      .encodeY("variance", Quant)
      .mark(Bar)

    renderHTML(plot, "variance")

  }

}
