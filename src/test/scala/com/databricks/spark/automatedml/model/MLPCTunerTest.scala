package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.{AbstractUnitSpec, AutomationUnitTestsUtil}
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
