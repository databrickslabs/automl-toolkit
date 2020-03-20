package com.databricks.labs.automl.utilities

object ValidationUtilities {

  def fieldCreationAssertion(expectedFields: Array[String],
                             generatedFieldNames: Array[String]): Unit = {

    assert(
      generatedFieldNames.forall(expectedFields.contains),
      "did not create any unexpected columns"
    )
    assert(
      expectedFields.forall(generatedFieldNames.contains),
      "creating the correct columns and retaining appropriate fields"
    )

  }

}
