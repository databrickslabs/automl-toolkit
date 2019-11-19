package com.databricks.labs.automl.feature

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

class FeatureInteraction(modelingType: String, retentionMode: String)
    extends FeatureInteractionBase {

  import ModelingType._
  import FieldEncodingType._
  import InteractionRetentionMode._

  private var _labelCol: String = "label"

  private var _fullDataEntropy: Double = 0.0

  private var _fullDataVariance: Double = 0.0

  private var _continuousDiscretizerBucketCount: Int = 10

  private var _parallelism: Int = 4

  private var _targetInteractionPercentage: Double = 25.0

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

  def setParallelism(value: Int): this.type = {
    require(
      value > 0,
      s"Parallelism value $value is invalid.  Must be 1 or greater."
    )
    _parallelism = value
    this
  }

  def setTargetInteractionPercentage(value: Double): this.type = {
    require(
      value > 0,
      s"Target Percentage allowance for inclusion must be a positive Double. $value is invalid."
    )
    _targetInteractionPercentage = value
    this
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

  /**
    * Private method for evaluating an interacted column
    * @param df A DataFrame to be used for candidate feature interaction evaluations
    * @param candidate The InteractionPayload for the parents of left/right to make the interacted feature
    * @param totalRecordCount Total number of records in the DataFrame (calculated only once for the Object)
    * @return InteractionResult payload of interaction scores associated with the interacted features
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def evaluateInteraction(df: DataFrame,
                                  candidate: InteractionPayload,
                                  totalRecordCount: Long): InteractionResult = {

    // Generate a subset DataFrame, create the interaction column, and retain only the fields needed.
    val evaluationDf = df
      .select(candidate.left, candidate.right, _labelCol)

    val interactedDf = interactProduct(evaluationDf, candidate)

    // Score the interaction
    val score = scoreColumn(
      interactedDf,
      getModelType(modelingType),
      candidate.outputName,
      getFieldType("continuous"),
      totalRecordCount
    )

    InteractionResult(
      candidate.left,
      candidate.right,
      candidate.outputName,
      score
    )

  }

  /**
    * Private method for comparing the parents scores to the interacted feature score and return a Boolean keep / not keep for the interacted feature in the final
    * data set and configuration for this module.
    * @param interactionResult the evaluated result of an interacted feature
    * @param leftScore left parent of interaction's score
    * @param rightScore right parent of interaction's score
    * @return Boolean value of whether to keep the interacted field or not
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def parentCompare(interactionResult: InteractionResult,
                            leftScore: Double,
                            rightScore: Double): Boolean = {

    val percentageChangeLeft =
      calculatePercentageChange(leftScore, interactionResult.score)

    val percentageChangeRight =
      calculatePercentageChange(rightScore, interactionResult.score)

    getRetentionMode(retentionMode) match {
      case Optimistic =>
        getModelType(modelingType) match {
          case Regressor =>
            percentageChangeLeft <= _targetInteractionPercentage | percentageChangeRight <= _targetInteractionPercentage
          case Classifier =>
            percentageChangeLeft >= -1 * _targetInteractionPercentage | percentageChangeRight >= -1 * _targetInteractionPercentage
        }
      case Strict =>
        getModelType(modelingType) match {
          case Regressor =>
            percentageChangeLeft <= _targetInteractionPercentage & percentageChangeRight <= _targetInteractionPercentage
          case Classifier =>
            percentageChangeLeft >= -1 * _targetInteractionPercentage & percentageChangeRight >= -1 * _targetInteractionPercentage
        }
    }

  }

  /**
    * Main method for generating a list of interaction candidates based on the configuration specified in the class configuration.
    * @param df The DataFrame to process interactions for
    * @param nominalFields The nominal fields (String Indexed) to be used for interaction
    * @param continuousFields The continuous fields (Original Numeric Types) to be used for interaction
    * @return Array[InteractionPayload] for candidate fields interactions that meet the acceptance criteria as set by configuration.
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  def generateCandidates(
    df: DataFrame,
    nominalFields: Array[String],
    continuousFields: Array[String]
  ): Array[InteractionPayload] = {

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

    val interactionCandidates = generateInteractionCandidates(
      nominalFields ++ continuousFields
    )

    val forkJoinTaskSupport = new ForkJoinTaskSupport(
      new ForkJoinPool(_parallelism)
    )
    val candidateChecks = interactionCandidates.par
    candidateChecks.tasksupport = forkJoinTaskSupport

    val scoredCandidates = candidateChecks.map { x =>
      evaluateInteraction(df, x, totalRecordCount)
    }.toArray

    var interactionBuffer = ArrayBuffer[InteractionPayload]()

    // Iterate over the evaluations and determine whether to keep them

    for (x <- scoredCandidates) {

      if (parentCompare(
            x,
            mergedParentScores(x.left),
            mergedParentScores(x.right)
          ))
        interactionBuffer += InteractionPayload(x.left, x.right, x.interaction)

    }

    interactionBuffer.toArray

  }

  /**
    * Method for determining feature interaction candidates, apply those candidates as new fields to the DataFrame,
    * and return a configuration payload that has the information about the interactions that can be used in a Pipeline.
    * @param df DataFrame to be used to calculate and potentially add feature interactions to
    * @param nominalFields Fields from the DataFrame that were originally non-numeric (Character, String, etc.)
    * @param continuousFields Fields from the DataFrame that were originally numeric, continuous types.
    * @return FeatureInteractionCollection -> the DataFrame with candidate feature interactions added in and the
    *         payload of interaction features and their constituent parents in order to recreate for a Pipeline.
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  def createCandidates(
    df: DataFrame,
    nominalFields: Array[String],
    continuousFields: Array[String]
  ): FeatureInteractionCollection = {

    val fieldsToCreate = generateCandidates(df, nominalFields, continuousFields)

    var data = df

    for (c <- fieldsToCreate) {
      data = interactProduct(data, c)
    }

    FeatureInteractionCollection(data, fieldsToCreate)

  }

}
