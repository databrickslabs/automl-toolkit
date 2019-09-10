package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.exceptions.MlFlowValidationException
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class MlFlowLoggingValidationStageTransformerTest extends AbstractUnitSpec{

  "MlFlowLoggingValidationStageTransformerTest" should "not changed input dataset" in {
    val spark = AutomationUnitTestsUtil.sparkSession
    val mlFlowLoggingValidationStageTransformer = new MlFlowLoggingValidationStageTransformer()
      .setMlFlowLoggingFlag(false)
      .setMlFlowExperimentName("test_name")
      .setMlFlowTrackingURI("test_Uri")
      .setMlFlowAPIToken("test_token")
    val adultDf = AutomationUnitTestsUtil.getAdultDf()
    val adultDfMlFlowValidation = mlFlowLoggingValidationStageTransformer.transform(adultDf)
    assert(adultDf.count() == adultDfMlFlowValidation.count(),
      "MlFlowLoggingValidationStageTransformerTest should not have changed input dataset rows")
    assert(adultDf.columns.length == adultDfMlFlowValidation.columns.length,
      "MlFlowLoggingValidationStageTransformerTest should not have changed number of columns")
    assert(adultDf.columns.sameElements(adultDfMlFlowValidation.columns),
      "MlFlowLoggingValidationStageTransformerTest should not have changed columns")
  }

  it should "throw MlFlowValidationException" in {
    a[MlFlowValidationException] should be thrownBy {
      new MlFlowLoggingValidationStageTransformer()
        .setMlFlowLoggingFlag(true)
        .setMlFlowExperimentName("test_name")
        .setMlFlowTrackingURI("test_Uri")
        .setMlFlowAPIToken("test_token")
        .transform(AutomationUnitTestsUtil.getAdultDf())
    }
  }
}
