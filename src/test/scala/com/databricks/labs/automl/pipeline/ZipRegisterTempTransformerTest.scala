package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}
import org.apache.log4j.{Level, LogManager}

class ZipRegisterTempTransformerTest extends AbstractUnitSpec {

  "ZipRegisterTempTransformer" should "add Id column, retain feature columns and create temp view of original dataset" in {
    val testVars = PipelineTestUtils.getTestVars()
    val featureColumns = Array("age_trimmed","workclass_trimmed","fnlwgt_trimmed","label")
    val zipRegisterTempTransformer = new ZipRegisterTempTransformer()
      .setFeatureColumns(featureColumns)
      .setLabelColumn(testVars.labelCol)
      .setTempViewOriginalDatasetName("zipRegisterTempTransformer")
      .setDebugEnabled(true)
//    LogManager.getRootLogger.setLevel(Level.DEBUG)
    val transformedAdultDf = zipRegisterTempTransformer.transform(testVars.df)
    assert(transformedAdultDf.count() == 99, "transformed table rows shouldn't have changed")
    assert(transformedAdultDf.columns.contains(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL), "Id column should have been generated")
  }

}
