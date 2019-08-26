package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class ZipRegisterTempTransformerTest extends AbstractUnitSpec {

  "ZipRegisterTempTransformer" should "add Id column, retain feature columns and create temp view of original dataset" in {
    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
    val featureColumns = Array("age_trimmed","workclass_trimmed","fnlwgt_trimmed","label")
    val zipRegisterTempTransformer =
      new ZipRegisterTempTransformer()
      .setFeatureColumns(featureColumns)
      .setTempViewOriginalDatasetName("zipRegisterTempTransformer")

    val transformedAdultDf = zipRegisterTempTransformer.transform(adultDfwithLabel)

    assert(transformedAdultDf.count() == 99, "transformed table rows shouldn't have changed")
    assert(transformedAdultDf.columns.contains(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL), "Id column should have been generated")


  }

}
