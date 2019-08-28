package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class DropColumnsTransformerTest extends AbstractUnitSpec {

  "DropColumnsTransformer" should "drop columns" in {

    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
    val columnsToRemove = Array("age_trimmed","workclass_trimmed","fnlwgt_trimmed")

    val dropColumnsTransformer = new DropColumnsTransformer()
      .setInputCols(columnsToRemove)

   val dfWithDroppedCols = dropColumnsTransformer.transform(adultDfwithLabel)

    assert(!dfWithDroppedCols.columns.exists(item => columnsToRemove.contains(item)),
    "DropColumnsTransformer should have removed input columns")

  }

}
