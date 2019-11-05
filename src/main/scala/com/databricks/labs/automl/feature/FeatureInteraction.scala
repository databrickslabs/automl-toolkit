package com.databricks.labs.automl.feature

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

case class VarianceData(labelValue: Double, variance: Double)
case class EntropyData(labelValue: Double, entropy: Double)
case class InteractionPayload(left: String, right: String, outputName: String)

trait FeatureInteractionBase {

  final val AGGREGATE_COLUMN: String = "totalCount"
  final val COUNT_COLUMN: String = "count"
  final val RATIO_COLUMN: String = "labelRatio"
  final val TOTAL_RATIO_COLUMN: String = "totalRatio"
  final val ENTROPY_COLUMN: String = "entropy"
  final val FIELD_ENTROPY_COLUMN: String = "fieldEntropy"
  final val QUANTILE_THRESHOLD: Double = 0.5
  final val QUANTILE_PRECISION: Double = 0.95
  final val VARIANCE_STATISTIC: String = "stddev"

}

class FeatureInteraction extends FeatureInteractionBase {

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

  def initialize = ???

}
