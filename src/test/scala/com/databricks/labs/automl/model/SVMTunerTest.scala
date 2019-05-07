package com.databricks.labs.automl.model

import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class SVMTunerTest extends AbstractUnitSpec {

  "SVMTuner" should "throw IllegalArgumentException for passing invalid params" in {
    a [IllegalArgumentException] should be thrownBy {
      new SVMTuner(null).evolveBest()
    }
  }

  it should "should throw IllegalArgumentException for passing invalid dataset" in {
    a [IllegalArgumentException] should be thrownBy {
      new SVMTuner(AutomationUnitTestsUtil.sparkSession.emptyDataFrame).evolveBest()
    }
  }
}
