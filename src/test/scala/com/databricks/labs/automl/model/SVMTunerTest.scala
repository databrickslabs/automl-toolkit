package com.databricks.labs.automl.model

import com.databricks.labs.automl.model.tools.split.DataSplitUtility
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class SVMTunerTest extends AbstractUnitSpec {

  "SVMTuner" should "throw IllegalArgumentException for passing invalid params" in {
    a[IllegalArgumentException] should be thrownBy {
      val splitData = DataSplitUtility.split(
        AutomationUnitTestsUtil.getAdultDf(),
        1,
        "random",
        "label",
        "dbfs:/test",
        "cache",
        "SVM",
        2,
        0.7,
        "synth",
        "datetime",
        0.02,
        0.6
      )

      new SVMTuner(null, splitData).evolveBest()
    }
  }

  it should "should throw IllegalArgumentException for passing invalid dataset" in {
    a[IllegalArgumentException] should be thrownBy {
      val splitData = DataSplitUtility.split(
        AutomationUnitTestsUtil.getAdultDf(),
        1,
        "random",
        "label",
        "dbfs:/test",
        "cache",
        "SVM",
        2,
        0.7,
        "synth",
        "datetime",
        0.02,
        0.6
      )

      new SVMTuner(
        AutomationUnitTestsUtil.sparkSession.emptyDataFrame,
        splitData
      ).evolveBest()
    }
  }

}
