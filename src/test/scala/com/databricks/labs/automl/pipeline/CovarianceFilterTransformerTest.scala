package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.{
  AbstractUnitSpec,
  AutomationUnitTestsUtil,
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

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    val sampleDatadf = Seq(
      Sample(1.0, 3.0, 0, 0, 1),
      Sample(1.0, 3.0, 2, 0, 2),
      Sample(2.0, 1.0, 2, 0, 3),
      Sample(1.0, 2.0, 4, 0, 1),
      Sample(2.0, 1.0, 6, 0, 2),
      Sample(1.0, 2.0, 8, 0, 3),
      Sample(2.0, 1.0, 9, 0, 4)
    ).toDF()

    sampleDatadf.show(10)

    val stages = new ArrayBuffer[PipelineStage]
    stages += new CovarianceFilterTransformer()
      .setFeatureColumns(Array("a", "b", "c"))
      .setLabelColumn("label")
      .setFeatureCol("features")
      .setCorrelationCutoffHigh(0.90)
      .setCorrelationCutoffLow(-0.50)

    val transformedDf = PipelineTestUtils
      .saveAndLoadPipeline(
        stages.toArray,
        sampleDatadf,
        "covar-filter-pipeline"
      )
      .transform(sampleDatadf)

    transformedDf.show(10)

    assert(
      !transformedDf.columns.exists(_.equals("b")),
      "Covariance filter should have removed column 'b'"
    )
  }
}
