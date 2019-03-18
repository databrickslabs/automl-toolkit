package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.{AbstractUnitSpec, AutomationUnitTestsUtil}

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
