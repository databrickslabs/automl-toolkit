package com.databricks.labs.automl.exploration.visualizations

import com.databricks.labs.automl.exploration.tools.OneDimStats
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}

class OneDimVisualizationsTest extends AbstractUnitSpec {

  it should "work" in {

    val data = DiscreteTestDataGenerator.generateDecayArray(100)

    val result = OneDimStats.evaluate(data)

    val render = OneDimVisualizations.generateOneDimPlots(result)

    println(render)
  }

}
