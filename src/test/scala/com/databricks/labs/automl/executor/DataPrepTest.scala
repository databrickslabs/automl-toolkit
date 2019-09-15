package com.databricks.labs.automl.executor

import com.databricks.labs.automl.params.DataGeneration
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}
import org.apache.spark.sql.types.{DataTypes, StructType}

class DataPrepTest extends AbstractUnitSpec {

  it should "throw NullPointerException for passing null dataset" in {
    a [NullPointerException] should be thrownBy {
      new DataPrep(null).prepData()
    }
  }

  it should "throw AssertionError for passing empty dataset" in {
    a [AssertionError] should be thrownBy {
      new DataPrep(AutomationUnitTestsUtil.sparkSession.emptyDataFrame).prepData()
    }
  }

  it should "return valid DataGeneration for preparing data" in {
    val adultDataset = AutomationUnitTestsUtil.getAdultDf()
    val dataGeneration: DataGeneration = new DataPrep(adultDataset).prepData()
    assert(dataGeneration != null, "DataPrep should not have returned null for a valid input dataset")
    assert(dataGeneration.data != null, "DataPrep should not have returned null dataset")
    assert(dataGeneration.fields != null, "DataPrep should not have returned null fields")
    assert(dataGeneration.modelType != null, "DataPrep should not have returned null model type")
    assert(dataGeneration.data.count() == adultDataset.count(), "DataPrep should not have returned different rows for input Dataset")
    assert(dataGeneration.fields.size == adultDataset.columns.size + 1, "DataPrep should not have changed number of columns")
  }

  it should "return valid schema for preparing data" in {
    val adultDataset = AutomationUnitTestsUtil.getAdultDf()
    val newSchema: StructType = new DataPrep(adultDataset).prepData().data.schema
    val originalSchema: StructType = adultDataset.schema

    for(field <- newSchema) {
      val fieldName = field.name
      if(originalSchema.fieldNames.contains(fieldName)) {
        assert(field.dataType == DataTypes.DoubleType || field.dataType == DataTypes.IntegerType, s"column $fieldName should have been indexed properly")
      }
    }
  }

}
