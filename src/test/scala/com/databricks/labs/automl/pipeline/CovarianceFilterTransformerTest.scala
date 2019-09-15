package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}
import org.apache.spark.ml.PipelineStage

import scala.collection.mutable.ArrayBuffer

class CovarianceFilterTransformerTest extends AbstractUnitSpec {

  "CovarianceFilterTransformerTest" should "apply the filter with right settings" in {
    a [AssertionError] should be thrownBy {
      val testVars = PipelineTestUtils.getTestVars()
      val stages = new ArrayBuffer[PipelineStage]
      val nonFeatureCols = Array(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL, testVars.labelCol)
      stages += PipelineTestUtils
        .addZipRegisterTmpTransformerStage(
          testVars.labelCol,
          testVars.features
        )
      stages += new CovarianceFilterTransformer()
        .setLabelColumn(testVars.labelCol)
        .setFeatureColumns(testVars.features)
        .setCorrelationCutoffLow(-0.9)
        .setCorrelationCutoffHigh(0.9)
      PipelineTestUtils
        .saveAndLoadPipeline(stages.toArray, testVars.df, "covar-filter-pipeline")
        .transform(testVars.df)
        .show(10)
    }
  }
}
