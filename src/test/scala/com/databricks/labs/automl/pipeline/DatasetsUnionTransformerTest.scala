package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}
import org.apache.spark.sql.functions._

class DatasetsUnionTransformerTest extends AbstractUnitSpec {

  "DatasetsUnionTransformerTest" should "correctly union DFs" in {
    val testVars = PipelineTestUtils.getTestVars()
    val df1 = testVars.df.withColumn(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL, monotonically_increasing_id())
    val df2 = testVars.df.withColumn(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL, monotonically_increasing_id())

    new RegisterTempTableTransformer()
      .setTempTableName("test_1")
      .setStatement("Select * from __THIS__")
      .transform(df1)

    val unionDf = new DatasetsUnionTransformer()
      .setUnionDatasetName("test_1")
      .transform(df2)

    assert(unionDf.count() == df1.count() + df2.count(),
      "DatasetsUnionTransformer did not correctly union the datasets")

  }

}
