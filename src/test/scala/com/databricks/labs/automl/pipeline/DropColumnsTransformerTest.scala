package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil, PipelineTestUtils}
import ml.combust.bundle.BundleFile
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.bundle.SparkBundleContext
import resource._
import ml.combust.mleap.spark.SparkSupport._

import scala.collection.mutable.ArrayBuffer

class DropColumnsTransformerTest extends AbstractUnitSpec {

  "DropColumnsTransformer" should "drop columns" in {

    val testVars = PipelineTestUtils.getTestVars()

    val stages = new ArrayBuffer[PipelineStage]
    val nonFeatureCols = Array(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL, testVars.labelCol)
    val columnsToRemove = Array("age_trimmed","workclass_trimmed","fnlwgt_trimmed")

    stages += PipelineTestUtils
      .addZipRegisterTmpTransformerStage(
        testVars.labelCol,
        testVars.df.columns.filterNot(item => nonFeatureCols.contains(item))
      )

    stages += new DropColumnsTransformer()
      .setInputCols(columnsToRemove)

    val pipelineModel = PipelineTestUtils
      .saveAndLoadPipeline(stages.toArray, testVars.df, "drop-columns-pipeline")

   val dfWithDroppedCols = pipelineModel.transform(testVars.df)

    assert(!dfWithDroppedCols.columns.exists(item => columnsToRemove.contains(item)),
    "DropColumnsTransformer should have removed input columns")
//    val pipelineSavePath = AutomationUnitTestsUtil.getProjectDir() + "/target/pipeline-tests"
//    // then serialize pipeline
//    val sbc = SparkBundleContext().withDataset(pipelineModel.transform(testVars.df))
//    for(bf <- managed(BundleFile(s"jar:file:$pipelineSavePath/drop-column-mleap.zip"))) {
//      pipelineModel.writeBundle.save(bf)(sbc).get
//    }

  }

}
