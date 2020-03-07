package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}
import org.apache.spark.sql.functions._

class RepartitionTransformerTest extends AbstractUnitSpec {

  "RepartitionTransformer" should "return correct repartitioned dataset" in {
    val testVars = PipelineTestUtils.getTestVars()
    val inputDf = testVars.df.withColumn("automl_internal_id", monotonically_increasing_id())
    val repartitionTransformer = new RepartitionTransformer()
      .setPartitionScaleFactor(2)
    val transformedDf = repartitionTransformer.transform(inputDf)
    assert(transformedDf.rdd.getNumPartitions == inputDf.rdd.getNumPartitions * 2,
      "DataFrame wasn't repartitioned as expected")
  }
}
