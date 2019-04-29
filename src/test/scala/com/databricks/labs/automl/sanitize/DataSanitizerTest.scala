package com.databricks.labs.automl.sanitize

import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}
import org.apache.spark.sql.DataFrame

class DataSanitizerTest extends AbstractUnitSpec {

  "DataSanitizerTest" should "throw NullPointerException for null input dataset" in {
    a [NullPointerException] should be thrownBy {
      new DataSanitizer(null).decideModel()
    }
  }

  it should "return clean data for valid input dataset" in {
    val adultDataset: DataFrame = AutomationUnitTestsUtil.getAdultDf()
      .withColumnRenamed("class","label")
    val cleanedData = new DataSanitizer(adultDataset).generateCleanData()
    assert(cleanedData != null, "clean data object should not be null")
    assert(cleanedData._1 != null, "cleaned dataset should not be null")
    assert(cleanedData._2 != null, "Nafillconfig should not be been null")
    assert(cleanedData._3 != null, "Model decided cannot be null")
    assert(cleanedData._3 equals "classifier", "Model decided should be classifier")
  }

}
