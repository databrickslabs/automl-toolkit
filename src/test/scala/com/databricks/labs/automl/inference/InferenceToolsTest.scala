package com.databricks.labs.automl.inference

import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil, InferenceUnitTestUtil}
import com.fasterxml.jackson.core.JsonParseException

class InferenceToolsTest extends AbstractUnitSpec {

  it should "return null inference payload internal state" in {
    val inferencePayload: InferencePayload = new InferenceTools{}.createInferencePayload(null, null, null)
    assert(inferencePayload.data == null, "should have returned null dataset")
    assert(inferencePayload.modelingColumns == null, "should have returned null modeling columns")
    assert(inferencePayload.allColumns == null, "should have returned null columns")
  }

  it should "return removal of columns" in {
    val columnRemoved = "age_trimmed"
    val originalInferencePayload = InferenceUnitTestUtil.generateInferencePayload()
    val inferencePayload: InferencePayload = new InferenceTools{}.removeArrayOfColumns(
      InferenceUnitTestUtil.generateInferencePayload(), Array(columnRemoved))
    assert(inferencePayload != null, "should not have returned null inference payload")
    assert(!inferencePayload.allColumns.contains(columnRemoved), s"should have removed column: $columnRemoved")
    val noOfColumns = originalInferencePayload.allColumns.size - 1
    assert(inferencePayload.allColumns.size == noOfColumns, s"should have returned $noOfColumns number of columns")
  }

  it should "return inference payload in json for null" in {
    val inferenceJsonReturn: InferenceJsonReturn = new InferenceTools{}.convertInferenceConfigToJson(null)
    assert(inferenceJsonReturn.prettyJson equals "null", "should have returned null prettyJson for invalid input")
    assert(inferenceJsonReturn.compactJson equals "null", "should have returned null compactJson for invalid input")
  }

  it should "raise JsonParseException due to deserialization of an illegal json" in {
    a [JsonParseException] should be thrownBy {
      new InferenceTools {}.convertJsonConfigToClass("error")
    }
  }

  it should "raise ArrayIndexOutOfBoundsException due to extraction of inference String from an empty dataframe" in {
    a [ArrayIndexOutOfBoundsException] should be thrownBy {
      new InferenceTools {}.extractInferenceJsonFromDataFrame(AutomationUnitTestsUtil.sparkSession.emptyDataFrame)
    }
  }

  it should "raise ArrayIndexOutOfBoundsException due to extraction of inference config from an empty dataframe" in {
    a [ArrayIndexOutOfBoundsException] should be thrownBy {
      new InferenceTools {}.extractInferenceConfigFromDataFrame(AutomationUnitTestsUtil.sparkSession.emptyDataFrame)
    }
  }

}
