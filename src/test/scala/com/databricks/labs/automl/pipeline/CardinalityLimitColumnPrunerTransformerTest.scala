package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil, PipelineTestUtils}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class CardinalityLimitColumnPrunerTransformerTest extends AbstractUnitSpec {

  "CardinalityLimitColumnPrunerTransformerTest" should " should check cardinality" in {
    val testVars = PipelineTestUtils.getTestVars()
    val stages = new ArrayBuffer[PipelineStage]
    val nonFeatureCols = Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL, testVars.labelCol)
    stages += PipelineTestUtils
      .addZipRegisterTmpTransformerStage(
        testVars.labelCol,
        testVars.df.columns.filterNot(item => nonFeatureCols.contains(item))
      )
    stages += new CardinalityLimitColumnPrunerTransformer()
      .setLabelColumn(testVars.labelCol)
      .setCardinalityLimit(2)
    val pipelineModel = PipelineTestUtils.saveAndLoadPipeline(stages.toArray, testVars.df, "card-limit-pipeline")
    val adultCadDf = pipelineModel.transform(testVars.df)
    assertCardinalityTest(adultCadDf)
    adultCadDf.show(10)
  }

  private def assertCardinalityTest(adultCadDf: DataFrame): Unit = {
    assert(adultCadDf.columns.exists(
      item => Array("sex_trimmed", "label").contains(item)),
      "CardinalityLimitColumnPrunerTransformer should have retained columns with a defined cardinality")
  }

}
