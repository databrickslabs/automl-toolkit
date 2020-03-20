package com.databricks.labs.automl.executor

import com.databricks.labs.automl.params.DataGeneration
import com.databricks.labs.automl.utilities.ValidationUtilities
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}
import org.apache.spark.sql.AnalysisException
import org.apache.spark.sql.types.{DataTypes, StructType}

class DataPrepTest extends AbstractUnitSpec {

  it should "throw NullPointerException for passing null dataset" in {
    a[NullPointerException] should be thrownBy {
      new DataPrep(null).prepData()
    }
  }

  it should "throw AnalysisException for passing empty dataset" in {
    a[AnalysisException] should be thrownBy {
      new DataPrep(AutomationUnitTestsUtil.sparkSession.emptyDataFrame)
        .prepData()
    }
  }

  it should "return valid DataGeneration for preparing data" in {
    val adultDataset = AutomationUnitTestsUtil.getAdultDf()
    val dataGeneration: DataGeneration = new DataPrep(adultDataset).prepData()

    val EXPECTED_MODEL_TYPE = "classifier"

    val EXPECTED_FIELDS = Array(
      "age_trimmed_si",
      "workclass_trimmed_si",
      "education_trimmed_si",
      "education-num_trimmed_si",
      "marital-status_trimmed_si",
      "occupation_trimmed_si",
      "relationship_trimmed_si",
      "race_trimmed_si",
      "sex_trimmed_si",
      "capital-gain_trimmed_si",
      "capital-loss_trimmed_si",
      "hours-per-week_trimmed_si",
      "native-country_trimmed_si",
      "features",
      "label"
    )

    println(
      s"Data Generation fields: ${dataGeneration.data.schema.names.mkString(", ")}"
    )

    assert(
      dataGeneration != null,
      "DataPrep should not have returned null for a valid input dataset"
    )
    assert(
      dataGeneration.data != null,
      "DataPrep should not have returned null dataset"
    )
    assert(
      dataGeneration.fields != null,
      "DataPrep should not have returned null fields"
    )
    assert(
      dataGeneration.modelType != null,
      "DataPrep should not have returned null model type"
    )
    assert(
      dataGeneration.data.count() == adultDataset.count(),
      "DataPrep should not have returned different rows for input Dataset"
    )
    assert(
      dataGeneration.fields.length == adultDataset.columns.length,
      "DataPrep should not have changed number of columns"
    )

    assert(
      dataGeneration.modelType == EXPECTED_MODEL_TYPE,
      "Should have detected correct model type"
    )

    ValidationUtilities.fieldCreationAssertion(
      EXPECTED_FIELDS,
      dataGeneration.data.schema.names
    )

    ValidationUtilities.fieldCreationAssertion(
      EXPECTED_FIELDS,
      dataGeneration.fields
    )

  }

  it should "return valid schema for preparing data" in {
    val adultDataset = AutomationUnitTestsUtil.getAdultDf()
    val newSchema: StructType =
      new DataPrep(adultDataset).prepData().data.schema
    val originalSchema: StructType = adultDataset.schema

    for (field <- newSchema) {
      val fieldName = field.name
      if (originalSchema.fieldNames.contains(fieldName)) {
        assert(
          field.dataType == DataTypes.DoubleType || field.dataType == DataTypes.IntegerType,
          s"column $fieldName should have been indexed properly"
        )
      }
    }
  }

}
