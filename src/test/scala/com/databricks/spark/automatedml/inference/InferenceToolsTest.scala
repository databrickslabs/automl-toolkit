package com.databricks.spark.automatedml.inference

import com.databricks.spark.automatedml.{AbstractUnitSpec, AutomationUnitTestsUtil, InferenceUnitTestUtil}
import com.fasterxml.jackson.core.JsonParseException

class InferenceToolsTest extends AbstractUnitSpec {

  it should "return null inference payload internal state" in {
    val inferencePayload: InferencePayload = new InferenceTools{}.createInferencePayload(null, null, null)
    assert(inferencePayload.data == null)
    assert(inferencePayload.modelingColumns == null)
    assert(inferencePayload.allColumns == null)
  }

  it should "return removal of columns" in {
    val columnRemoved = "age_trimmed"
    val originalInferencePayload = InferenceUnitTestUtil.generateInferencePayload()
    val inferencePayload: InferencePayload = new InferenceTools{}.removeArrayOfColumns(
      InferenceUnitTestUtil.generateInferencePayload(), Array(columnRemoved))
    assert(inferencePayload != null)
    assert(!inferencePayload.allColumns.contains(columnRemoved))
    assert(inferencePayload.allColumns.size == originalInferencePayload.allColumns.size - 1)
  }

  it should "return inference payload in json for null" in {
    val inferenceJsonReturn: InferenceJsonReturn = new InferenceTools{}.convertInferenceConfigToJson(null)
    assert(inferenceJsonReturn.prettyJson equals "null")
    assert(inferenceJsonReturn.compactJson equals "null")
  }

  it should "raise JsonParseException due to deserialization of an illegal json" in {
    a [JsonParseException] should be thrownBy {
      val inferenceJsonReturn: InferenceMainConfig = new InferenceTools {}.convertJsonConfigToClass("error")
    }
  }

  it should "raise ArrayIndexOutOfBoundsException due to extraction of inference String from an empty dataframe" in {
    a [ArrayIndexOutOfBoundsException] should be thrownBy {
      val inferenceJsonFromDataframe: String = new InferenceTools {}.extractInferenceJsonFromDataFrame(AutomationUnitTestsUtil.sparkSession.emptyDataFrame)
    }
  }

  it should "raise ArrayIndexOutOfBoundsException due to extraction of inference config from an empty dataframe" in {
    a [ArrayIndexOutOfBoundsException] should be thrownBy {
      val inferenceMainConfig: InferenceMainConfig = new InferenceTools {}.extractInferenceConfigFromDataFrame(AutomationUnitTestsUtil.sparkSession.emptyDataFrame)
    }
  }

}
