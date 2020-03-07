package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}
import org.apache.spark.ml.PipelineStage

import scala.collection.mutable.ArrayBuffer

class OutlierFilterTransformerTest extends AbstractUnitSpec {

  it should "correctly apply outlier filtering" in {
    val testVars = PipelineTestUtils.getTestVars()
    val stages = new ArrayBuffer[PipelineStage]
    val nonFeatureCols =
      Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL, testVars.labelCol)
    stages += PipelineTestUtils
      .addZipRegisterTmpTransformerStage(
        testVars.labelCol,
        testVars.df.columns.filterNot(item => nonFeatureCols.contains(item))
      )
    stages ++= PipelineTestUtils.buildFeaturesPipelineStages(
      testVars.df,
      testVars.labelCol,
      dropColumns = false
    )
    stages += new DropColumnsTransformer()
      .setInputCols(Array(testVars.featuresCol))
    stages += new OutlierFilterTransformer()
      .setLabelColumn(testVars.labelCol)
      .setFilterBounds("both")
      .setLowerFilterNTile(0.1)
      .setUpperFilterNTile(0.4)
      .setFilterPrecision(0.01)
      .setContinuousDataThreshold(50)
      .setParallelism(10)
      .setFieldsToIgnore(Array.empty)
      .setDebugEnabled(false)

    val outlierDf = PipelineTestUtils
      .saveAndLoadPipeline(
        stages.toArray,
        testVars.df,
        "outlier-filter-pipeline"
      )
      .transform(testVars.df)
    outlierDf.show()
    assert(
      outlierDf.count() == 31,
      "OutlierFilterTransformer should have filtered rows, check outlier filter settings"
    )
  }

}
