package com.databricks.labs.automl.feature

import com.databricks.labs.automl.exceptions.ModelingTypeException
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

trait FeatureInteractionBase {
  import ModelingType._
  import FieldEncodingType._
  import InteractionRetentionMode._

  private final val allowableModelTypes = Array("classifier", "regressor")
  private final val allowableFieldTypes = Array("nominal", "continuous")
  private final val allowableRetentionModes = Array("optimistic", "strict")

  final val AGGREGATE_COLUMN: String = "totalCount"
  final val COUNT_COLUMN: String = "count"
  final val RATIO_COLUMN: String = "labelRatio"
  final val TOTAL_RATIO_COLUMN: String = "totalRatio"
  final val ENTROPY_COLUMN: String = "entropy"
  final val FIELD_ENTROPY_COLUMN: String = "fieldEntropy"
  final val QUANTILE_THRESHOLD: Double = 0.5
  final val QUANTILE_PRECISION: Double = 0.95
  final val VARIANCE_STATISTIC: String = "stddev"

  protected[feature] def getModelType(
    modelingType: String
  ): ModelingType.Value = {
    modelingType match {
      case "regressor"  => Regressor
      case "classifier" => Classifier
      case _            => throw ModelingTypeException(modelingType, allowableModelTypes)
    }
  }

  protected[feature] def getFieldType(
    fieldType: String
  ): FieldEncodingType.Value = {
    fieldType match {
      case "nominal"    => Nominal
      case "continuous" => Continuous
      case _            => throw ModelingTypeException(fieldType, allowableFieldTypes)
    }
  }

  protected[feature] def getRetentionMode(
    retentionMode: String
  ): InteractionRetentionMode.Value = {
    retentionMode match {
      case "optimistic" => Optimistic
      case "strict"     => Strict
      case _ =>
        throw ModelingTypeException(retentionMode, allowableRetentionModes)
    }
  }

  /**
    * Method for generating a collection of Interaction Candidates to be tested and applied to the feature set
    * if the tests for inclusion pass.
    * @param featureColumns List of the columns that make up the feature vector
    * @return Array of InteractionPayload values.
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  protected[feature] def generateInteractionCandidates(
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
    * Method for evaluating the percentage change to the score metric to normalize.
    * @param before Score of a parent feature
    * @param after Score of an interaction feature
    * @return the percentage change
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  protected[feature] def calculatePercentageChange(before: Double,
                                                   after: Double): Double = {

    (after - before) / math.abs(before) * 100.0

  }

  /**
    * Method for generating a product interaction between feature columns
    * @param df A DataFrame to add a field for an interaction between two columns
    * @param candidate InteractionPayload information about the two parent columns and the name of the new interaction column to be created.
    * @return A modified DataFrame with the new column.
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  protected[feature] def interactProduct(
    df: DataFrame,
    candidate: InteractionPayload
  ): DataFrame = {

    df.withColumn(
      candidate.outputName,
      col(candidate.left) * col(candidate.right)
    )

  }

}

case class VarianceData(labelValue: Double, variance: Double)
case class EntropyData(labelValue: Double, entropy: Double)
case class InteractionPayload(left: String, right: String, outputName: String)
case class InteractionResult(left: String,
                             right: String,
                             interaction: String,
                             score: Double)
case class FeatureInteractionCollection(
  data: DataFrame,
  interactionPayload: Array[InteractionPayload]
)

object ModelingType extends Enumeration {
  val Regressor = ModelType("regressor")
  val Classifier = ModelType("classifier")
  protected case class ModelType(modelType: String) extends super.Val()
  implicit def convert(value: Value): ModelType = value.asInstanceOf[ModelType]
}

object FieldEncodingType extends Enumeration {
  val Nominal = FieldType("nominal")
  val Continuous = FieldType("continuous")
  protected case class FieldType(fieldType: String) extends super.Val()
  implicit def convert(value: Value): FieldType = value.asInstanceOf[FieldType]
}

object InteractionRetentionMode extends Enumeration {
  val Optimistic = RetentionMode("optimistic")
  val Strict = RetentionMode("strict")
  protected case class RetentionMode(retentionMode: String) extends super.Val()
  implicit def convert(value: Value): RetentionMode =
    value.asInstanceOf[RetentionMode]
}
