package com.databricks.labs.automl.feature

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool
import com.databricks.labs.automl.feature.structures._
import com.databricks.labs.automl.pipeline.{
  DropColumnsTransformer,
  InteractionTransformer
}
import org.apache.spark.ml.Pipeline

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
    * @since 0.6.2
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
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def setFullDataVariance(df: DataFrame): this.type = {

    _fullDataVariance = scala.math.pow(
      df.select(_labelCol)
        .summary(VARIANCE_STATISTIC)
        .first()
        .getAs[String](_labelCol)
        .toDouble,
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
    * @since 0.6.2
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
    * @since 0.6.2
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
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def parentCompare(interactionResult: InteractionResult,
                            leftScore: ColumnScoreData,
                            rightScore: ColumnScoreData): Boolean = {

    val percentageChangeLeft =
      calculatePercentageChange(leftScore.score, interactionResult.score)

    val percentageChangeRight =
      calculatePercentageChange(rightScore.score, interactionResult.score)

    val keepCheck = getRetentionMode(retentionMode) match {
      case Optimistic =>
        getModelType(modelingType) match {
          case Regressor =>
            percentageChangeLeft <= _targetInteractionPercentage * -100 | percentageChangeRight <= _targetInteractionPercentage * -100
          case Classifier =>
            percentageChangeLeft >= _targetInteractionPercentage * -100 | percentageChangeRight >= _targetInteractionPercentage * -100
        }
      case Strict =>
        getModelType(modelingType) match {
          case Regressor =>
            percentageChangeLeft <= _targetInteractionPercentage * -100 & percentageChangeRight <= _targetInteractionPercentage * -100
          case Classifier =>
            percentageChangeLeft >= _targetInteractionPercentage * -100 & percentageChangeRight >= _targetInteractionPercentage * -100
        }
      case All => true
    }

    keepCheck

  }

  /**
    * Main method for generating a list of interaction candidates based on the configuration specified in the class configuration.
    * @param df The DataFrame to process interactions for
    * @param nominalFields The nominal fields (String Indexed) to be used for interaction
    * @param continuousFields The continuous fields (Original Numeric Types) to be used for interaction
    * @return Array[InteractionPayload] for candidate fields interactions that meet the acceptance criteria as set by configuration.
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateCandidates(
    df: DataFrame,
    nominalFields: Array[String],
    continuousFields: Array[String]
  ): Array[InteractionPayloadExtract] = {

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
      val interaction = evaluateInteraction(df, x, totalRecordCount)
      scoredCandidates += interaction
    }

    var interactionBuffer = ArrayBuffer[InteractionPayloadExtract]()

    // Iterate over the evaluations and determine whether to keep them

    for (x <- scoredCandidates) {

      if (parentCompare(
            x,
            mergedParentScores(x.left),
            mergedParentScores(x.right)
          ))
        interactionBuffer += InteractionPayloadExtract(
          x.left,
          mergedParentScores(x.left).dataType,
          x.right,
          mergedParentScores(x.right).dataType,
          x.interaction,
          x.score
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
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def createCandidates(
    df: DataFrame,
    nominalFields: Array[String],
    continuousFields: Array[String]
  ): FeatureInteractionCollection = {

    val fieldsToCreatePrime =
      generateCandidates(df, nominalFields, continuousFields)
    val fieldsToCreate = fieldsToCreatePrime.map(x => {
      InteractionPayload(
        x.left,
        x.leftDataType,
        x.right,
        x.rightDataType,
        x.outputName
      )
    })

    var data = df

    for (c <- fieldsToCreate) {
      data = interactProduct(data, c)
    }

    FeatureInteractionCollection(data, fieldsToCreatePrime)

  }

  /**
    * Method for generating interaction candidates and re-building a feature vector
    * @param df DataFrame to interact features with (that has a feature vector already built)
    * @param nominalFields Array of column names for nominal (string indexed) values
    * @param continuousFields Array of column names for continuous numeric values
    * @param featureVectorColumn Name of the feature vector column
    * @return DataFrame with a re-built feature vector that includes the interacted feature columns as part of it.
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def createCandidatesAndAddToVector(
    df: DataFrame,
    nominalFields: Array[String],
    continuousFields: Array[String],
    featureVectorColumn: String
  ): FeatureInteractionOutputPayload = {

    FeatureEvaluator.extractAndValidateSchema(df.schema, featureVectorColumn)

    val strippedDf = df.drop(featureVectorColumn)

    val candidatePayload =
      createCandidates(strippedDf, nominalFields, continuousFields)

    // Reset the nominal interaction fields
    val indexedInteractions = generateNominalIndexesInteractionFields(
      candidatePayload
    )

    // Build the Vector again
    val vectorFields = nominalFields ++ continuousFields

    val assemblerOutput = regenerateFeatureVector(
      indexedInteractions.data,
      vectorFields,
      indexedInteractions.adjustedFields,
      featureVectorColumn
    )

    val outputData = restructureSchema(
      assemblerOutput.data,
      df.schema.names,
      vectorFields,
      indexedInteractions.adjustedFields,
      featureVectorColumn,
      _labelCol
    )

    FeatureInteractionOutputPayload(
      outputData,
      vectorFields ++ indexedInteractions.adjustedFields,
      candidatePayload.interactionPayload
    )

  }

  /**
    * Method for generating a pipeline-friendly feature interaction to support serialization of the automl pipeline
    * properly.  Utilizes the InteractionTransformer to generate the fields required for inference
    * @param df DataFrame to be used for generating the interaction candidates and pipeline
    * @param nominalFields Nominal type numeric fields that are part of the vector
    * @param continuousFields Continuous type numeric fields that are part of the vector
    * @param featureVectorColumn Name of the current feature vector column
    * @return PipelineInteractionOutput which contains the pipeline to be applied to the automl pipeline flow.
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def createPipeline(df: DataFrame,
                     nominalFields: Array[String],
                     continuousFields: Array[String],
                     featureVectorColumn: String): PipelineInteractionOutput = {

    FeatureEvaluator.extractAndValidateSchema(df.schema, featureVectorColumn)

    // Create the pipeline stage for dropping the feature vector
    val columnDropTransformer =
      new DropColumnsTransformer().setInputCols(Array(featureVectorColumn))

    // Remove the feature vector
    val strippedDf = columnDropTransformer.transform(df)

    // Get the fields that are needed for interaction, if any
    val candidatePayload =
      createCandidates(strippedDf, nominalFields, continuousFields)

    // Create the fields through the Interaction Transformer
    val leftColumns = candidatePayload.interactionPayload.map(_.left)
    val rightColumns = candidatePayload.interactionPayload.map(_.right)

    val interactor = new InteractionTransformer()
      .setLeftCols(leftColumns)
      .setRightCols(rightColumns)

    // Create the string indexers
    val indexedInteractions = generateNominalIndexesInteractionFields(
      candidatePayload
    )

    val preIndexerFieldsToRemove = indexedInteractions.fieldsToRemove

    val indexedColumnDropTransformer =
      new DropColumnsTransformer().setInputCols(preIndexerFieldsToRemove)

    // Create the vector
    val vectorFields = nominalFields ++ continuousFields

    val assemblerOutput = regenerateFeatureVector(
      indexedInteractions.data,
      vectorFields,
      indexedInteractions.adjustedFields,
      featureVectorColumn
    )

    // create the pipeline
    val pipelineElement = new Pipeline().setStages(
      Array(columnDropTransformer) ++ Array(interactor) ++ indexedInteractions.indexers ++ Array(
        indexedColumnDropTransformer
      ) //++ Array(assemblerOutput.assembler)
    )

    PipelineInteractionOutput(
      pipelineElement,
      pipelineElement.fit(df).transform(df),
      vectorFields ++ indexedInteractions.adjustedFields,
      candidatePayload.interactionPayload
    )

  }

  /**
    * Private method for enforcing re-ordering of the Dataframe that is returned to preserve the structure of the
    * original dataframe before being passed to this module and to create appropriate placement of interacted features
    * @param data DataFrame that has been interacted
    * @param originalSchemaNames Names within the original schema
    * @param originalFeatureNames Features that were originally contained in the vector prior to interaction
    * @param interactedFields Fields that have been retained as interaction candidates
    * @param featureCol feature column name
    * @param labelCol label column name
    * @return DataFrame in correct order
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def restructureSchema(data: DataFrame,
                                originalSchemaNames: Array[String],
                                originalFeatureNames: Array[String],
                                interactedFields: Array[String],
                                featureCol: String,
                                labelCol: String): DataFrame = {

    val startingFields = originalFeatureNames
      .filterNot(x => x.contains(featureCol))
      .filterNot(x => x.contains(labelCol))

    val ignoredFields = originalSchemaNames
      .filterNot(originalFeatureNames.contains)
      .filterNot(x => x.contains(labelCol))
      .filterNot(x => x.contains(featureCol))
    val featureOrdered = startingFields.filterNot(ignoredFields.contains) ++ interactedFields
    val orderedFields = featureOrdered ++ ignoredFields ++ Array(
      featureCol,
      labelCol
    )
    data.select(orderedFields map col: _*)

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
  ): Array[InteractionPayloadExtract] = {
    new FeatureInteraction(modelingType, retentionMode)
      .setLabelCol(labelCol)
      .setContinuousDiscretizerBucketCount(continuousDiscretizerBucketCount)
      .setParallelism(parallelism)
      .setTargetInteractionPercentage(targetInteractionPercentage)
      .generateCandidates(data, nominalFields, continuousFields)

  }

  def interactionPipeline(
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
  ): PipelineInteractionOutput = {
    new FeatureInteraction(modelingType, retentionMode)
      .setLabelCol(labelCol)
      .setContinuousDiscretizerBucketCount(continuousDiscretizerBucketCount)
      .setParallelism(parallelism)
      .setTargetInteractionPercentage(targetInteractionPercentage)
      .createPipeline(data, nominalFields, continuousFields, featureCol)

  }

}
