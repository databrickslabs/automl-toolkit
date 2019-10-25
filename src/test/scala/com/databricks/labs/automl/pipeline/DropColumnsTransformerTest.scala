package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil, PipelineTestUtils}
import ml.combust.bundle.BundleFile
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.bundle.SparkBundleContext
import resource._
import ml.combust.mleap.spark.SparkSupport._

import scala.collection.mutable.ArrayBuffer

class DropColumnsTransformerTest extends AbstractUnitSpec {

//  "DropColumnsTransformer" should "drop columns" in {
//    val testVars = PipelineTestUtils.getTestVars()
//    val stages = new ArrayBuffer[PipelineStage]
//    val nonFeatureCols = Array(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL, testVars.labelCol)
//    val columnsToRemove = Array("age_trimmed","workclass_trimmed","fnlwgt_trimmed")
//    stages += PipelineTestUtils
//      .addZipRegisterTmpTransformerStage(
//        testVars.labelCol,
//        testVars.df.columns.filterNot(item => nonFeatureCols.contains(item))
//      )
//    stages += new DropColumnsTransformer()
//      .setInputCols(columnsToRemove)
//    val pipelineModel = PipelineTestUtils
//      .saveAndLoadPipeline(stages.toArray, testVars.df, "drop-columns-pipeline")
//    val dfWithDroppedCols = pipelineModel.transform(testVars.df)
//    assert(!dfWithDroppedCols.columns.exists(item => columnsToRemove.contains(item)),
//      "DropColumnsTransformer should have removed input columns")
//  }

  "DropColumnsTransformer" should "drop columns with Broadcast" in {
    val testVars = PipelineTestUtils.getTestVars()
    val stages = new ArrayBuffer[PipelineStage]
    val nonFeatureCols = Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL, testVars.labelCol)
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
    val bc = testVars.df.sparkSession.sparkContext.broadcast(pipelineModel)
    val dfWithDroppedCols = bc.value.transform(testVars.df)
    assert(!dfWithDroppedCols.columns.exists(item => columnsToRemove.contains(item)),
      "DropColumnsTransformer should have removed input columns")
  }
}
