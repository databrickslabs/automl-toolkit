package com.databricks.labs.automl.feature

import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, lit, log2, sum, variance}
import com.databricks.labs.automl.feature.structures._
import org.apache.spark.sql.types.StructType

object FeatureEvaluator extends FeatureInteractionBase {

  /**
    * Helper method for calculating the Information Gain of a feature field
    * @param df DataFrame that contains at least the fieldToTest and the Label Column
    * @param fieldToTest The field to calculate Information Gain for
    * @param totalRecordCount Total number of records in the data set
    * @return The Information Gain of the field
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  def calculateCategoricalInformationGain(df: DataFrame,
                                          labelColumn: String,
                                          fieldToTest: String,
                                          totalRecordCount: Long): Double = {

    val groupedData = df.groupBy(labelColumn, fieldToTest).count()

    val fieldCounts = df
      .select(fieldToTest)
      .groupBy(fieldToTest)
      .count()
      .withColumnRenamed(COUNT_COLUMN, AGGREGATE_COLUMN)

    val mergeCounts = groupedData
      .join(fieldCounts, Seq(fieldToTest), "left")
      .withColumn(RATIO_COLUMN, col(COUNT_COLUMN) / col(AGGREGATE_COLUMN))
      .withColumn(
        ENTROPY_COLUMN,
        lit(-1) * col(RATIO_COLUMN) * log2(col(RATIO_COLUMN))
      )
      .withColumn(
        TOTAL_RATIO_COLUMN,
        col(AGGREGATE_COLUMN) / lit(totalRecordCount)
      )

    val distinctValues =
      mergeCounts.select(fieldToTest, TOTAL_RATIO_COLUMN).distinct()

    val mergedEntropy = mergeCounts
      .groupBy(fieldToTest)
      .agg(sum(ENTROPY_COLUMN).alias(ENTROPY_COLUMN))

    val joinedEntropy = mergedEntropy
      .join(distinctValues, Seq(fieldToTest))
      .withColumn(
        FIELD_ENTROPY_COLUMN,
        col(ENTROPY_COLUMN) * col(TOTAL_RATIO_COLUMN)
      )
      .select(fieldToTest, FIELD_ENTROPY_COLUMN)
      .collect()
      .map(r => EntropyData(r.get(0).toString.toDouble, r.getDouble(1)))

    joinedEntropy.map(_.entropy).sum / joinedEntropy.length.toDouble

  }

  /**
    * Helper method for converting a continuous feature to a discrete bucketed value so that entropy can be calculated
    * effectively for the feature.
    * @param df DataFrame containing at least the field to test in continuous numeric format
    * @param fieldToTest The name of the field under conversion
    * @return A Dataframe with the continuous value converted to a quantized bucket membership value.
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  def discretizeContinuousFeature(df: DataFrame,
                                  fieldToTest: String,
                                  bucketCount: Int): DataFrame = {

    val renamedFieldToTest = s"d_$fieldToTest"

    val discretizer = new QuantileDiscretizer()
      .setInputCol(renamedFieldToTest)
      .setOutputCol(fieldToTest)
      .setNumBuckets(bucketCount)
      .setHandleInvalid("keep")
      .setRelativeError(QUANTILE_PRECISION)

    val modifiedData = df.withColumnRenamed(fieldToTest, renamedFieldToTest)

    discretizer.fit(modifiedData).transform(modifiedData)

  }

  /**
    * Helper method for handling Information Gain Calculation for classification data set when dealing with continuous
    * (numeric) feature elements.  The continuous feature will be split upon the configured value of _continuousDiscretizerBucketCount,
    * which is set by overriding .setContinuousDiscretizerBucketCount(<Int>)
    * @param df DataFrame that contains the feature to test and the label column
    * @param fieldToTest The feature field that is under test for entropy evaluation
    * @param totalRecordCount Total number of elements in the data set.
    * @return Information Gain associated with the feature field based on splits that could occur.
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  def calculateContinuousInformationGain(df: DataFrame,
                                         labelCol: String,
                                         fieldToTest: String,
                                         totalRecordCount: Long,
                                         bucketCount: Int): Double = {

    val adjustedFieldData =
      discretizeContinuousFeature(df, fieldToTest, bucketCount)

    calculateCategoricalInformationGain(
      adjustedFieldData,
      labelCol,
      fieldToTest,
      totalRecordCount
    )

  }

  /**
    * Method for calculating the variance of a categorical (nominal) field based on a post-split first-layer variance
    * of the label column's values to determine the minimum variance achievable in the label column.
    * @param df DataFrame that contains the label column and the field under test for minimum by-group variance
    * @param labelColumn The label column of the data set
    * @param fieldToTest The feature column to test for variance reduction
    * @return The minimum split variance of the aggregated label data by nominal group of the fieldToTest
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  def calculateCategoricalVariance(df: DataFrame,
                                   labelColumn: String,
                                   fieldToTest: String): Double = {

    val groupedData = df
      .select(labelColumn, fieldToTest)
      .groupBy(fieldToTest)
      .agg(variance(fieldToTest).alias(fieldToTest))
      .collect()
      .map(r => VarianceData(r.get(0).toString.toDouble, r.getDouble(1)))
    groupedData.map(_.variance).min
  }

  /**
    * Method for calculating the variance of a continuous field for variance reduction in the label column based on
    * bucketized grouping of the field under test.
    * @param df DataFrame that contains the label column and the field under test of continuous numeric type
    * @param labelColumn The label column of the data set
    * @param fieldToTest The field to test (continuous numeric) that need to be evaluated
    * @param bucketCount The number of quantized buckets to create to group the field under test into in order to
    *                    simulate where a decision split would occur.
    * @return The minimum split variance of each of the buckets that have been created
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  def calculateContinuousVariance(df: DataFrame,
                                  labelColumn: String,
                                  fieldToTest: String,
                                  bucketCount: Int): Double = {

    val convertedContinuousData =
      discretizeContinuousFeature(df, fieldToTest, bucketCount)

    calculateCategoricalVariance(
      convertedContinuousData,
      labelColumn,
      fieldToTest
    )

  }

  /**
    * Helper method for extracting field names and ensuring that the feature vector is present
    * @param schema Schema of the DataFrame undergoing feature interaction
    * @param featureVector The name of the features column
    * @return Array of column names of the DataFrame
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def extractAndValidateSchema(schema: StructType,
                               featureVector: String): Unit = {

    val schemaFields = schema.names

    require(
      schemaFields.contains(featureVector),
      s"The feature vector column $featureVector does not " +
        s"exist in the DataFrame supplied to FeatureInteraction.createCandidatesAndAddToVector.  Field listing is: " +
        s"${schemaFields.mkString(", ")} "
    )

  }

}
