package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil, PipelineTestUtils}
import org.apache.spark.sql.DataFrame

class CardinalityLimitColumnPrunerTransformerTest extends AbstractUnitSpec {

  "CardinalityLimitColumnPrunerTransformerTest" should " should check cardinality" in {

    val testVars = PipelineTestUtils.getTestVars()

    val cardinalityLimitColumnPrunerTransformer = new CardinalityLimitColumnPrunerTransformer()
      .setLabelColumn(testVars.labelCol)
      .setCardinalityLimit(2)

    val adultCadDf = cardinalityLimitColumnPrunerTransformer.transform(testVars.df)

    assertCardinalityTest(adultCadDf)

    assertCardinalityTest(
      PipelineTestUtils
        .saveAndLoadPipeline(Array(cardinalityLimitColumnPrunerTransformer), testVars.df, "card-limit-trans-pipe")
        .transform(testVars.df)
    )
  }

  private def assertCardinalityTest(adultCadDf: DataFrame): Unit = {
    assert(adultCadDf.columns.exists(
      item => Array("sex_trimmed", "label").contains(item)),
      "CardinalityLimitColumnPrunerTransformer should have retained columns with a defined cardinality")
  }

}
