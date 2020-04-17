package com.databricks.labs.automl.exploration.tools

import com.databricks.labs.automl.AbstractUnitSpec
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}

class PCAReducerTest extends AbstractUnitSpec {

  final private val data = getData
  final private val label = "label"
  final private val feature = "features"
  final private val pcaFeature = "pcaFeatures"

  lazy val sparkSession: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("providentiaml-unit-tests")
    .getOrCreate()

  sparkSession.sparkContext.setLogLevel("FATAL")

  final private val ADULT_SCHEMA = StructType(
    Array(
      StructField("age", IntegerType, nullable = false),
      StructField("workclass", StringType, nullable = false),
      StructField("finalweight", DoubleType, nullable = false),
      StructField("education", StringType, nullable = false),
      StructField("education-num-years", IntegerType, nullable = false),
      StructField("marital_status", StringType, nullable = false),
      StructField("occupation", StringType, nullable = false),
      StructField("relationship", StringType, nullable = false),
      StructField("race", StringType, nullable = false),
      StructField("sex", StringType, nullable = false),
      StructField("capital_gain", DoubleType, nullable = false),
      StructField("capital_loss", DoubleType, nullable = false),
      StructField("hours_per_week", IntegerType, nullable = false),
      StructField("native_country", StringType, nullable = false),
      StructField("class", StringType, nullable = false)
    )
  )

  private def convertCsvToDf(csvPath: String): DataFrame = {
    sparkSession.read
      .format("csv")
      .option("header", value = true)
      .schema(ADULT_SCHEMA)
      .load(getClass.getResource(csvPath).getPath)
  }

  private def getData: DataFrame = {
    import sparkSession.implicits._
    val adultDf = convertCsvToDf("/adult_data.csv")

    adultDf.schema.names
      .foldLeft(adultDf) { case (df, i) => df.withColumn(i, trim(col(i))) }
      .withColumn("label", when($"class" === "<=50K", 0).otherwise(1))
      .drop("class")
  }

  it should "get PCA correctly from raw data with column retention mode" in {

    val SCALER = "minMax"

    val result =
      PCAReducer(label, feature, pcaFeature, SCALER).withConvertedColumns
        .executePipeline(data)

    assert(
      !result.pcEigenDataFrame.isEmpty,
      s"The EigenVector DataFrame is empty"
    )
    assert(
      result.pcMatrix.length == data.schema.names.length - 1,
      s"The Principle Components eigen values" +
        s"were not calculated properly"
    )
    assert(
      result.data.schema.names.length == 35,
      s"The returned data does not have the correct number of " +
        s"columns.  Columns returned are: ${result.data.schema.names.mkString(", ")}"
    )
    assert(!result.explainedVariances.isEmpty, "Explained Variances are empty.")

  }

  it should "get PCA correctly from raw data with original column mode" in {

    val SCALER = "minMax"

    val result =
      PCAReducer(label, feature, pcaFeature, SCALER).withOriginalColumns
        .executePipeline(data)

    assert(
      !result.pcEigenDataFrame.isEmpty,
      s"The EigenVector DataFrame is empty"
    )
    assert(
      result.pcMatrix.length == data.schema.names.length - 1,
      s"The Principle Components eigen values" +
        s"were not calculated properly"
    )
    assert(
      result.data.schema.names.length == 17,
      s"The returned data does not have the correct number of " +
        s"columns.  Columns returned are: ${result.data.schema.names.mkString(", ")}"
    )
    assert(!result.explainedVariances.isEmpty, "Explained Variances are empty.")

  }

  it should "get PCA correctly from raw data with original column mode and non-default scaling" in {

    val result =
      PCAReducer(label, feature, pcaFeature).withOriginalColumns.withStandardScaling
        .executePipeline(data)

    assert(
      !result.pcEigenDataFrame.isEmpty,
      s"The EigenVector DataFrame is empty"
    )
    assert(
      result.pcMatrix.length == data.schema.names.length - 1,
      s"The Principle Components eigen values" +
        s"were not calculated properly"
    )
    assert(
      result.data.schema.names.length == 17,
      s"The returned data does not have the correct number of " +
        s"columns.  Columns returned are: ${result.data.schema.names.mkString(", ")}"
    )
    assert(!result.explainedVariances.isEmpty, "Explained Variances are empty.")

  }

  it should "throw an exception for improper scaler settings" in {

    a[AssertionError] should be thrownBy {

      val SCALER = "notImplemented"
      PCAReducer(label, feature, pcaFeature, SCALER)

    }

  }

  it should "throw an exception for improperly configured label column specification" in {

    a[AssertionError] should be thrownBy {
      val SCALER = "minMax"
      val BAD_LABEL = "notInData"

      val result =
        PCAReducer(BAD_LABEL, feature, pcaFeature, SCALER).executePipeline(data)

    }

  }

}
