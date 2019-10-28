package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}
import org.apache.spark.ml.PipelineStage

import scala.collection.mutable.ArrayBuffer

class PearsonFilterTransformerTest extends AbstractUnitSpec {

  "PearsonFilterTransformerTest" should "correctly apply pearson filter" in {
    val testVars = PipelineTestUtils.getTestVars()
    val stages = new ArrayBuffer[PipelineStage]()
    val nonFeatureCols =
      Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL, testVars.labelCol)
    stages += PipelineTestUtils
      .addZipRegisterTmpTransformerStage(
        testVars.labelCol,
        testVars.df.columns.filterNot(item => nonFeatureCols.contains(item))
      )
    val vectFeatures = PipelineTestUtils
      .getVectorizedFeatures(
        testVars.df,
        testVars.labelCol,
        Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
      )
    stages ++= PipelineTestUtils
      .buildFeaturesPipelineStages(
        testVars.df,
        testVars.labelCol,
        dropColumns = false,
        ignoreCols = Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
      )
    stages += new PearsonFilterTransformer()
      .setLabelColumn(testVars.labelCol)
      .setFeatureCol(testVars.featuresCol)
      .setFeatureColumns(vectFeatures)
      .setAutoFilterNTile(0.75)
      .setFilterDirection("greater")
      .setFilterManualValue(0)
      .setFilterMode("auto")
      .setFilterStatistic("pearsonStat")
    val pearsonDf = PipelineTestUtils
      .saveAndLoadPipeline(
        stages.toArray,
        testVars.df,
        "pearson-filter-pipeline"
      )
      .transform(testVars.df)
    assert(
      pearsonDf.columns.length == 6,
      "PearsonFilterTransformer should have retained only 6 columns"
    )
  }
}
