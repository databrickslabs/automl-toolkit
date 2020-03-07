package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}
import org.apache.spark.sql.DataFrame

class AutoMlOutputDatasetTransformerTest extends AbstractUnitSpec {

  "AutoMlOutputDatasetTransformer" should "drop Id column, retain original columns from the original dataset" in {
    val testVars = PipelineTestUtils.getTestVars()
    val pipelineAdultDf = new ZipRegisterTempTransformer()
      .setFeatureColumns(testVars.features)
      .setTempViewOriginalDatasetName(testVars.tempTableName)
      .setLabelColumn(testVars.labelCol)
      .transform(testVars.df)
    val pipelineOutputDf = new AutoMlOutputDatasetTransformer()
      .setFeatureColumns(testVars.features)
      .setTempViewOriginalDatasetName(testVars.tempTableName)
      .setLabelColumn(testVars.labelCol)
      .transform(pipelineAdultDf)
    assertAutoMlOutputDatasetTransformerTest(pipelineAdultDf, pipelineOutputDf)
  }

  "AutoMlOutputDatasetTransformer" should "work with Pipeline save/load" in {
    val testVars = PipelineTestUtils.getTestVars()
    val pipelineAdultDf = new ZipRegisterTempTransformer()
      .setFeatureColumns(testVars.features)
      .setTempViewOriginalDatasetName(testVars.tempTableName)
      .setLabelColumn(testVars.labelCol)
    val pipelineOutputDf = new AutoMlOutputDatasetTransformer()
      .setFeatureColumns(testVars.features)
      .setTempViewOriginalDatasetName(testVars.tempTableName)
      .setLabelColumn(testVars.labelCol)
   val transformedAdultDfwithLabel =  PipelineTestUtils
     .saveAndLoadPipeline(Array(pipelineAdultDf, pipelineOutputDf), testVars.df, "automl-output-df-pipe")
       .transform(testVars.df)
    assert(!transformedAdultDfwithLabel.columns.contains(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL),
      "AutoMlOutputDatasetTransformer should have dropped Id column and retained original columns")
  }


  def assertAutoMlOutputDatasetTransformerTest(pipelineAdultDf: DataFrame,
                                               pipelineOutputDf: DataFrame): Unit = {
    assert(pipelineAdultDf.columns.contains(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL),
      "ZipRegisterTempTransformer stage should have added Id column")
    assert(!pipelineOutputDf.columns.contains(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL),
      "AutoMlOutputDatasetTransformer should have dropped Id column and retained original columns")
  }
}
