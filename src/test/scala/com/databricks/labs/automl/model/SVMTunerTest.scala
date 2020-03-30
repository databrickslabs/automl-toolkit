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
        "income",
        "dbfs:/test",
        "cache",
        "SVM"
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
        "income",
        "dbfs:/test",
        "cache",
        "SVM"
      )

      new SVMTuner(
        AutomationUnitTestsUtil.sparkSession.emptyDataFrame,
        splitData
      ).evolveBest()
    }
  }

}
