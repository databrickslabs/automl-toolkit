package com.databricks.labs.automl.feature

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class FeatureInteraction(modelingType: String) extends FeatureInteractionBase {

  import ModelingType._
  import FieldEncodingType._

  private var _labelCol: String = "label"

  private var _fullDataEntropy: Double = 0.0

  private var _fullDataVariance: Double = 0.0

  private var _continuousDiscretizerBucketCount: Int = 10

  def setLabelCol(value: String): this.type = {
    _labelCol = value
    this
  }

  def setContinuousDiscretizerBucketCount(value: Int): this.type = {

    require(
      value > 1,
      s"Continuous Discretizer Bucket Count for continuous features must be greater than 1. $value is invalid."
    )
    _continuousDiscretizerBucketCount = value
    this
  }

  /**
    * Private method for generating a collection of Interaction Candidates to be tested and applied to the feature set
    * if the tests for inclusion pass.
    * @param featureColumns List of the columns that make up the feature vector
    * @return Array of InteractionPayload values.
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def generateInteractionCandidates(
    featureColumns: Array[String]
  ): Array[InteractionPayload] = {
    val colIdx = featureColumns.zipWithIndex
    colIdx.flatMap {
      case (x, i) =>
        val maxIdx = colIdx.length
        for (j <- Range(i + 1, maxIdx)) yield {
          InteractionPayload(x, colIdx(j)._1, s"i_${x}_${colIdx(j)._1}")
        }
    }
  }

  /**
    * Helper method to set the class property for data-level entropy based on the values of a nominal label column
    * @param df The raw data frame
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def setFullDataEntropy(df: DataFrame): this.type = {

    val uniqueLabelEntries = df.select(_labelCol).groupBy(_labelCol).count()

    val labelEntropy =
      uniqueLabelEntries
        .agg(sum(COUNT_COLUMN).alias(AGGREGATE_COLUMN))
        .crossJoin(uniqueLabelEntries)
        .withColumn(RATIO_COLUMN, col(COUNT_COLUMN) / col(AGGREGATE_COLUMN))
        .withColumn(
          ENTROPY_COLUMN,
          lit(-1) * col(RATIO_COLUMN) * log2(col(RATIO_COLUMN))
        )
        .select(_labelCol, ENTROPY_COLUMN)
        .collect()
        .map(r => EntropyData(r.get(0).toString.toDouble, r.getDouble(1)))

    _fullDataEntropy = labelEntropy.map(_.entropy).sum
    this

  }

  /**
    * Private method for setting the data set label's variance value
    * @param df The source DataFrame
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def setFullDataVariance(df: DataFrame): this.type = {

    _fullDataVariance = scala.math.pow(
      df.select(_labelCol)
        .summary(VARIANCE_STATISTIC)
        .collect()(0)
        .getDouble(0),
      2
    )
    this
  }

  /**
    * Private method for scoring a column based on the model type and the field type
    * @param df Dataframe for evaluation
    * @param modelType Model Type: Either Classifier or Regressor from Enum
    * @param fieldToTest The field to be scored
    * @param fieldType The type of the field: Either Nominal (String Indexed) or Continuous from Enum
    * @param totalRecordCount Total number of rows in the data set in order to calculate Entropy correctly
    * @return A Score as Double
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def scoreColumn(df: DataFrame,
                          modelType: ModelingType.Value,
                          fieldToTest: String,
                          fieldType: FieldEncodingType.Value,
                          totalRecordCount: Long): Double = {

    val subsetData = df.select(fieldToTest, _labelCol)

    modelType match {
      case Classifier =>
        fieldType match {
          case Nominal =>
            FeatureEvaluator.calculateCategoricalInformationGain(
              subsetData,
              _labelCol,
              fieldToTest,
              totalRecordCount
            )
          case Continuous =>
            FeatureEvaluator.calculateContinuousInformationGain(
              subsetData,
              _labelCol,
              fieldToTest,
              totalRecordCount,
              _continuousDiscretizerBucketCount
            )
        }
      case Regressor =>
        fieldType match {
          case Nominal =>
            FeatureEvaluator.calculateCategoricalVariance(
              subsetData,
              _labelCol,
              fieldToTest
            )
          case Continuous =>
            FeatureEvaluator.calculateContinuousVariance(
              subsetData,
              _labelCol,
              fieldToTest,
              _continuousDiscretizerBucketCount
            )
        }
    }

  }

  def initialize(df: DataFrame,
                 nominalFields: Array[String],
                 continuousFields: Array[String]) = {

    val modelType = getModelType(modelingType)

    val totalRecordCount = df.count()

    modelType match {
      case Regressor  => setFullDataVariance(df)
      case Classifier => setFullDataEntropy(df)
    }

    val nominalScores = nominalFields.map { x =>
      x -> scoreColumn(
        df,
        modelType,
        x,
        getFieldType("nominal"),
        totalRecordCount
      )

    }.toMap

    val continuousScores = continuousFields.map { x =>
      x -> scoreColumn(
        df,
        modelType,
        x,
        getFieldType("continuous"),
        totalRecordCount
      )
    }.toMap

    val mergedParentScores = nominalScores ++ continuousScores

    //TODO:

    // do interaction mapping and add fields

    // compare fields to parents, drop if below / above threshold as configured.

  }

}
