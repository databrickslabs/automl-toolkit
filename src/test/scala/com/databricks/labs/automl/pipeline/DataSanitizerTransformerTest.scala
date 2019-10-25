package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}
import org.apache.spark.ml.PipelineStage

import scala.collection.mutable.ArrayBuffer

class DataSanitizerTransformerTest extends AbstractUnitSpec {

  "DataSanitizerTransformer" should " sanitize based on the settings" in {
    val testVars = PipelineTestUtils.getTestVars()
    val stages = new ArrayBuffer[PipelineStage]
    val nonFeatureCols = Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL, testVars.labelCol)
    stages += PipelineTestUtils
      .addZipRegisterTmpTransformerStage(
        testVars.labelCol,
        testVars.df.columns.filterNot(item => nonFeatureCols.contains(item))
      )
    stages += new DataSanitizerTransformer()
      .setLabelColumn(testVars.labelCol)
      .setFeatureCol("features")
      .setNaFillFlag(true)
    stages ++= PipelineTestUtils.buildFeaturesPipelineStages(testVars.df, testVars.labelCol)
    val transformedAdultDf = PipelineTestUtils
      .saveAndLoadPipeline(stages.toArray, testVars.df, "data_sanitizer_stage")
      .transform(testVars.df)
    assert(transformedAdultDf.count() == testVars.df.count(), "Number of rows shouldn't have changed")
    assert(transformedAdultDf.columns.length == 3, "Should only contain label, ID and feature columns")
    transformedAdultDf.show(10)
  }

}
