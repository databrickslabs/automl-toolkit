package com.databricks.labs.automl.model

import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}
import org.apache.spark.sql.AnalysisException

class MLPCTunerTest extends AbstractUnitSpec {

  "MLPCTuner" should "throw NullPointerException for passing invalid params" in {
    a [NullPointerException] should be thrownBy {
      new MLPCTuner(null).evolveBest()
    }
  }

  it should "should throw AnalysisException for passing invalid dataset" in {
    a [AnalysisException] should be thrownBy {
      new MLPCTuner(AutomationUnitTestsUtil.sparkSession.emptyDataFrame).evolveBest()
    }
  }
}
