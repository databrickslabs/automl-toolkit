package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}

class ColumnNameTransformerTest extends AbstractUnitSpec {

  "ColumnNameTransformerTest" should "remove columns" in {
    val testVars = PipelineTestUtils.getTestVars()
    val columnNameTransformer = new ColumnNameTransformer()
      .setInputColumns(Array("age_trimmed"))
      .setOutputColumns(Array("age_trimmed_r"))
    val renamedDf = columnNameTransformer.transform(testVars.df)
    assert(renamedDf.count() == testVars.df.count(), "ColumnNameTransformerTest should not have changed number of rows")
    assert(renamedDf.columns.length == testVars.df.columns.length, "ColumnNameTransformerTest should not have changed number of columns")
    assert(renamedDf.columns.contains("age_trimmed_r"), "ColumnNameTransformerTest should contain renamed column")
    assert(!renamedDf.columns.contains("age_trimmed"), "ColumnNameTransformerTest should not contain original column")
  }
}
