package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.{
  AbstractUnitSpec,
  DiscreteTestDataGenerator,
  PipelineTestUtils
}
import org.apache.spark.ml.PipelineStage

import scala.collection.mutable.ArrayBuffer

case class Sample(a: Double,
                  b: Double,
                  c: Double,
                  label: Int,
                  automl_internal_id: Long)

class CovarianceFilterTransformerTest extends AbstractUnitSpec {

  "CovarianceFilterTransformerTest" should "apply the filter with right settings" in {

    val EXPECTED_REMAINING_COLS =
      Array("a1", "c2", "d1", "d2", "label", "automl_internal_id")

    val data = DiscreteTestDataGenerator.generateFeatureCorrelationData(1000)

    val stages = new ArrayBuffer[PipelineStage]
    stages += new CovarianceFilterTransformer()
      .setFeatureColumns(Array("a", "b", "c"))
      .setLabelColumn("label")
      .setFeatureCol("features")
      .setCorrelationCutoffHigh(1.0)
      .setCorrelationCutoffLow(-1.0)

    val transformedDf = PipelineTestUtils
      .saveAndLoadPipeline(stages.toArray, data._1, "covar-filter-pipeline")
      .transform(data._1)

    transformedDf.show(10)

    assert(
      EXPECTED_REMAINING_COLS.forall(transformedDf.schema.names.contains),
      "kept correct columns"
    )
    assert(
      transformedDf.schema.names.forall(EXPECTED_REMAINING_COLS.contains),
      "removed correct columns"
    )
  }
}
