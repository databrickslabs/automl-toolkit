package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}
import org.apache.spark.ml.PipelineStage

import scala.collection.mutable.ArrayBuffer

class VarianceFilterTransformerTest extends AbstractUnitSpec {

  "VarianceFilterTransformerTest" should "apply the filter correctly" in {
    val testVars = PipelineTestUtils.getTestVars()
    val stages = new ArrayBuffer[PipelineStage]
    stages += PipelineTestUtils
      .addZipRegisterTmpTransformerStage(
        testVars.labelCol,
        testVars.df.columns.filterNot(item => AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL.equals(item))
      )
    stages ++= PipelineTestUtils.buildFeaturesPipelineStages(testVars.df, testVars.labelCol, dropColumns = false)
    stages += new VarianceFilterTransformer()
      .setLabelColumn(testVars.labelCol)
      .setFeatureCol(testVars.featuresCol)
    PipelineTestUtils
      .saveAndLoadPipeline(stages.toArray, testVars.df, "variance-filter-pipeline")
      .transform(testVars.df).show(10)
  }
}
