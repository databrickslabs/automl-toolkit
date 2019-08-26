package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class AutoMlOutputDatasetTransformerTest extends AbstractUnitSpec {

  "AutoMlOutputDatasetTransformer" should "drop Id column, retain original columns from the original dataset" in {
    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
    val featureColumns = Array("age_trimmed","workclass_trimmed","fnlwgt_trimmed","label")
    val tempViewName = "zipRegisterTempTransformer_1"


    val pipelineAdultDf = new ZipRegisterTempTransformer()
      .setFeatureColumns(featureColumns)
      .setTempViewOriginalDatasetName(tempViewName)
      .transform(adultDfwithLabel)

    val pipelineOutputDf = new AutoMlOutputDatasetTransformer()
      .setFeatureColumns(featureColumns)
      .setTempViewOriginalDatasetName(tempViewName)
      .transform(pipelineAdultDf)

    assert(pipelineAdultDf.columns.contains(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL),
      "ZipRegisterTempTransformer stage should have added Id column")
    assert(!pipelineOutputDf.columns.contains(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL),
      "AutoMlOutputDatasetTransformer should have dropped Id column and retained original columns")

  }
}
