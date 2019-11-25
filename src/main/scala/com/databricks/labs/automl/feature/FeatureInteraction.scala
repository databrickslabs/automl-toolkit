package com.databricks.labs.automl.feature

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool
import com.databricks.labs.automl.feature.structures._

class FeatureInteraction(modelingType: String, retentionMode: String)
    extends FeatureInteractionBase {

  import com.databricks.labs.automl.feature.structures.FieldEncodingType._
  import com.databricks.labs.automl.feature.structures.InteractionRetentionMode._
  import com.databricks.labs.automl.feature.structures.ModelingType._

  private var _labelCol: String = "label"

  private var _fullDataEntropy: Double = 0.0

  private var _fullDataVariance: Double = 0.0

  private var _continuousDiscretizerBucketCount: Int = 10

  private var _parallelism: Int = 4

  private var _targetInteractionPercentage: Double = modelingType match {
    case "regressor"  => -1.0
    case "classifier" => 1.0
  }

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

    val dataModelDecision =
      (candidate.leftDataType, candidate.rightDataType) match {
        case ("nominal", "nominal") => "nominal"
        case _                      => "continuous"
      }

    // Score the interaction
    val score = scoreColumn(
      interactedDf,
      getModelType(modelingType),
      candidate.outputName,
      getFieldType(dataModelDecision),
      totalRecordCount
    )

    // DEBUG
    println(s"Score for ${candidate.outputName}: $score")

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
                            leftScore: ColumnScoreData,
                            rightScore: ColumnScoreData): Boolean = {

    val percentageChangeLeft =
      calculatePercentageChange(leftScore.score, interactionResult.score)

    val percentageChangeRight =
      calculatePercentageChange(rightScore.score, interactionResult.score)

    // DEBUG
    println(
      s"Percentage Change from ${interactionResult.left}: $percentageChangeLeft for ${interactionResult.interaction} with score: ${interactionResult.score} compared to $leftScore"
    )
    println(
      s"Percentage Change from ${interactionResult.right}: $percentageChangeRight for ${interactionResult.interaction} with score: ${interactionResult.score} compared to $rightScore"
    )

    val keepCheck = getRetentionMode(retentionMode) match {
      case Optimistic =>
        getModelType(modelingType) match {
          case Regressor =>
            percentageChangeLeft <= _targetInteractionPercentage * -1 | percentageChangeRight <= _targetInteractionPercentage * -1
          case Classifier =>
            percentageChangeLeft >= _targetInteractionPercentage | percentageChangeRight >= _targetInteractionPercentage
        }
      case Strict =>
        getModelType(modelingType) match {
          case Regressor =>
            percentageChangeLeft <= _targetInteractionPercentage * -1 & percentageChangeRight <= _targetInteractionPercentage * -1
          case Classifier =>
            percentageChangeLeft >= _targetInteractionPercentage & percentageChangeRight >= _targetInteractionPercentage
        }
    }

    // DEBUG
    println(s"Decision to keep this interaction is: ${keepCheck.toString}")

    keepCheck

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
      x -> ColumnScoreData(
        scoreColumn(
          df,
          modelType,
          x,
          getFieldType("nominal"),
          totalRecordCount
        ),
        "nominal"
      )

    }.toMap

    val continuousScores = continuousFields.map { x =>
      x -> ColumnScoreData(
        scoreColumn(
          df,
          modelType,
          x,
          getFieldType("continuous"),
          totalRecordCount
        ),
        "continuous"
      )
    }.toMap

    val mergedParentScores = nominalScores ++ continuousScores

    val interactionCandidatePayload = nominalFields.map(
      x => ColumnTypeData(x, "nominal")
    ) ++ continuousFields.map(y => ColumnTypeData(y, "continuous"))

    val interactionCandidates = generateInteractionCandidates(
      interactionCandidatePayload
    )

    val forkJoinTaskSupport = new ForkJoinTaskSupport(
      new ForkJoinPool(_parallelism)
    )

    val scoredCandidates = ArrayBuffer[InteractionResult]()

    val candidateChecks = interactionCandidates.par
    candidateChecks.tasksupport = forkJoinTaskSupport

    candidateChecks.foreach { x =>
      scoredCandidates += evaluateInteraction(df, x, totalRecordCount)
    }

    var interactionBuffer = ArrayBuffer[InteractionPayload]()

    // Iterate over the evaluations and determine whether to keep them

    for (x <- scoredCandidates) {

      if (parentCompare(
            x,
            mergedParentScores(x.left),
            mergedParentScores(x.right)
          ))
        interactionBuffer += InteractionPayload(
          x.left,
          mergedParentScores(x.left).dataType,
          x.right,
          mergedParentScores(x.right).dataType,
          x.interaction
        )

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

  /**
    * Method for generating interaction candidates and re-building a feature vector
    * @param df DataFrame to interact features with (that has a feature vector already built)
    * @param nominalFields Array of column names for nominal (string indexed) values
    * @param continuousFields Array of column names for continuous numeric values
    * @param featureVectorColumn Name of the feature vector column
    * @return DataFrame with a re-built feature vector that includes the interacted feature columns as part of it.
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  def createCandidatesAndAddToVector(df: DataFrame,
                                     nominalFields: Array[String],
                                     continuousFields: Array[String],
                                     featureVectorColumn: String) = {

    val currentFields = df.schema.names

    require(
      currentFields.contains(featureVectorColumn),
      s"The feature vector column $featureVectorColumn does not " +
        s"exist in the DataFrame supplied to FeatureInteraction.createCandidatesAndAddToVector.  Field listing is: " +
        s"${currentFields.mkString(", ")} "
    )

    val strippedDf = df.drop(featureVectorColumn)

    val candidatePayload =
      createCandidates(strippedDf, nominalFields, continuousFields)

    // Reset the nominal interaction fields
    val indexedInteractions = generateNominalIndexesInteractionFields(
      candidatePayload
    )

    // Build the Vector again
    val vectorFields = nominalFields ++ continuousFields

    FeatureInteractionOutputPayload(
      regenerateFeatureVector(
        indexedInteractions.data,
        vectorFields,
        indexedInteractions.adjustedFields,
        featureVectorColumn
      ),
      vectorFields ++ indexedInteractions.adjustedFields
    )

  }

}

object FeatureInteraction {

  def interactFeatures(
    data: DataFrame,
    nominalFields: Array[String],
    continuousFields: Array[String],
    modelingType: String,
    retentionMode: String,
    labelCol: String,
    featureCol: String,
    continuousDiscretizerBucketCount: Int,
    parallelism: Int,
    targetInteractionPercentage: Double
  ): FeatureInteractionOutputPayload =
    new FeatureInteraction(modelingType, retentionMode)
      .setLabelCol(labelCol)
      .setContinuousDiscretizerBucketCount(continuousDiscretizerBucketCount)
      .setParallelism(parallelism)
      .setTargetInteractionPercentage(targetInteractionPercentage)
      .createCandidatesAndAddToVector(
        data,
        nominalFields,
        continuousFields,
        featureCol
      )

  def interactDataFrame(
    data: DataFrame,
    nominalFields: Array[String],
    continuousFields: Array[String],
    modelingType: String,
    retentionMode: String,
    labelCol: String,
    continuousDiscretizerBucketCount: Int,
    parallelism: Int,
    targetInteractionPercentage: Double
  ): FeatureInteractionCollection = {
    new FeatureInteraction(modelingType, retentionMode)
      .setLabelCol(labelCol)
      .setContinuousDiscretizerBucketCount(continuousDiscretizerBucketCount)
      .setParallelism(parallelism)
      .setTargetInteractionPercentage(targetInteractionPercentage)
      .createCandidates(data, nominalFields, continuousFields)
  }

  def interactionReport(
    data: DataFrame,
    nominalFields: Array[String],
    continuousFields: Array[String],
    modelingType: String,
    retentionMode: String,
    labelCol: String,
    continuousDiscretizerBucketCount: Int,
    parallelism: Int,
    targetInteractionPercentage: Double
  ): Array[InteractionPayload] = {
    new FeatureInteraction(modelingType, retentionMode)
      .setLabelCol(labelCol)
      .setContinuousDiscretizerBucketCount(continuousDiscretizerBucketCount)
      .setParallelism(parallelism)
      .setTargetInteractionPercentage(targetInteractionPercentage)
      .generateCandidates(data, nominalFields, continuousFields)

  }

}
