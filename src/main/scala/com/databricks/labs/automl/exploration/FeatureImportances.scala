package com.databricks.labs.automl.exploration

import com.databricks.labs.automl.exploration.structures.{
  FeatureImportanceConfig,
  FeatureImportanceOutput,
  FeatureImportanceReturn,
  FeatureImportanceTools
}
import com.databricks.labs.automl.feature.FeatureInteraction
import com.databricks.labs.automl.model.{RandomForestTuner, XGBoostTuner}
import com.databricks.labs.automl.pipeline.FeaturePipeline
import com.databricks.labs.automl.sanitize.DataSanitizer
import com.databricks.labs.automl.utils.SparkSessionWrapper
import ml.dmlc.xgboost4j.scala.spark.{
  XGBoostClassificationModel,
  XGBoostRegressionModel
}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  *
  * @param data DataFrame: A data to determine which fields are most important when used to predict a label column.
  * @param config FeatureImportanceConfig: A configuration object for the Feature Importance run.
  * @param cutoffType String: The type of cutoff to use, either: 'count', 'none', or 'value' <br>
  *                     @note Count => Return the top n most important fields maxing the naming of the original data,
  *                           in descending order, are returned.<br>
  *                           None => A sorted list of columns, in descending order of importance, are returned.<br>
  *                           Value => All values above the thresholded value set in cutoffValue are returned
  *                           in descending order.
  * @param cutoffValue Double: Linked to cutoffType, providing a threshold value for how many fields to return.
  *                    @note for cutoffType 'None', this value can be set to 0.0
  * @example ```
  *
  *          val genericMapOverrides = Map("labelCol" -> "label", "tunerKFold" -> 2, "tunerTrainSplitMethod" ->
  *          "stratified", "tunerNumberOfGenerations" -> 4, "tunerNumberOfMutationsPerGeneration" -> 6,
  *          "tunerInitialGenerationPermutationCount" -> 25,
  *           "fieldsToIgnoreInVector" -> Array("final_weight"),"tunerInitialGenerationMode" -> "permutations")
  *
  *           val xgbConfig = ConfigurationGenerator.generateConfigFromMap("XGBoost", "classifier", genericMapOverrides)
  *
  *           val featConfig = ConfigurationGenerator.generateFeatureImportanceConfig(xgbConfig)
  *
  *           val importances = new FeatureImportances(data, featConfig, "count", 5.0).generateFeatureImportances()
  *          ```
  */
class FeatureImportances(data: DataFrame,
                         config: FeatureImportanceConfig,
                         cutoffType: String,
                         cutoffValue: Double)
    extends FeatureImportanceTools
    with SparkSessionWrapper {

  import com.databricks.labs.automl.exploration.structures.FeatureImportanceModelFamily._
  import com.databricks.labs.automl.exploration.structures.ModelType._
  import com.databricks.labs.automl.exploration.structures.CutoffTypes._

  private val cutOff = cutoffTypeEvaluator(cutoffType)
  private val modelFamily = featureImportanceFamilyEvaluator(
    config.featureImportanceModelFamily
  )
  private val modelType = modelTypeEvaluator(config.modelType)

  val importanceCol = "Importance"
  val featureCol = "Feature"

  private def fillNaValues(): DataFrame = {

    val (cleanedData, fillMap, modelDetectedType) = new DataSanitizer(data)
      .setLabelCol(config.labelCol)
      .setFeatureCol(config.featuresCol)
      .setNumericFillStat(config.numericFillStat)
      .setCharacterFillStat(config.characterFillStat)
      .setModelSelectionDistinctThreshold(
        config.modelSelectionDistinctThreshold
      )
      .setFieldsToIgnoreInVector(config.fieldsToIgnore)
      .setParallelism(config.dataPrepParallelism)
      .setFilterPrecision(0.01)
      .generateCleanData()

    cleanedData
  }

  private def createFeatureVector(df: DataFrame): FeatureImportanceOutput = {

    val (pipelinedData, vectorFields, totalFields) = new FeaturePipeline(df)
      .setLabelCol(config.labelCol)
      .setFeatureCol(config.featuresCol)
      .setDateTimeConversionType(config.dateTimeConversionType)
      .makeFeaturePipeline(config.fieldsToIgnore)

    new FeatureImportanceOutput {
      override def data: DataFrame = pipelinedData

      override def fieldsInVector: Array[String] = vectorFields

      override def allFields: Array[String] = totalFields
    }

  }

  /**
    * Private method for interacting features to determine if they'll be useful in a model
    * @param vectorPayload DataFrame payload with feature vector already created
    * @return FeatureImportanceOutput with the DataFrame supporting potential additions to the feature vector
    */
  private def interactFeatures(
    vectorPayload: FeatureImportanceOutput
  ): FeatureImportanceOutput = {

    val nominalFeatures = vectorPayload.fieldsInVector
      .filter(x => x.takeRight(3) == "_si")
      .filterNot(x => x.contains(config.labelCol))

    val continuousFeatures = vectorPayload.fieldsInVector
      .diff(nominalFeatures)
      .filterNot(_.contains(config.labelCol))
      .filterNot(_.contains(config.featuresCol))

    val interaction = FeatureInteraction.interactFeatures(
      vectorPayload.data,
      nominalFeatures,
      continuousFeatures,
      config.modelType,
      config.featureInteractionRetentionMode,
      config.labelCol,
      config.featuresCol,
      config.featureInteractionContinuousDiscretizerBucketCount,
      config.featureInteractionParallelism,
      config.featureInteractionTargetInteractionPercentage
    )

    new FeatureImportanceOutput {
      override def data: DataFrame = interaction.data

      override def fieldsInVector: Array[String] =
        interaction.fullFeatureVectorColumns

      override def allFields: Array[String] =
        interaction.data.schema.names
          .filterNot(_.contains(config.labelCol))
          .filterNot(_.contains(config.featuresCol))
    }

  }

  private def cleanFieldNames(fields: Array[String]): Array[String] = {
    fields.map { x =>
      x.takeRight(3) match {
        case "_si" => x.dropRight(3)
        case "_oh" => x.dropRight(3)
        case _     => x
      }
    }
  }

  private def getImportances(
    df: DataFrame,
    vectorFields: Array[String]
  ): Map[String, Double] = {

    val adjustedFieldNames = cleanFieldNames(vectorFields)

    val result = modelFamily match {
      case RandomForest =>
        val rfModel = new RandomForestTuner(df, config.modelType)
          .setLabelCol(config.labelCol)
          .setFeaturesCol(config.featuresCol)
          .setRandomForestNumericBoundaries(config.numericBoundaries)
          .setRandomForestStringBoundaries(config.stringBoundaries)
          .setScoringMetric(config.scoringMetric)
          .setTrainPortion(config.trainPortion)
          .setTrainSplitMethod(config.trainSplitMethod)
          .setTrainSplitChronologicalColumn(
            config.trainSplitChronologicalColumn
          )
          .setTrainSplitChronologicalRandomPercentage(
            config.trainSplitChronlogicalRandomPercentage
          )
          .setParallelism(config.parallelism)
          .setKFold(config.kFold)
          .setSeed(config.seed)
          .setOptimizationStrategy(config.scoringOptimizationStrategy)
          .setFirstGenerationGenePool(config.firstGenerationGenePool)
          .setNumberOfMutationGenerations(config.numberOfGenerations)
          .setNumberOfMutationsPerGeneration(
            config.numberOfMutationsPerGeneration
          )
          .setNumberOfParentsToRetain(config.numberOfParentsToRetain)
          .setGeneticMixing(config.geneticMixing)
          .setGenerationalMutationStrategy(config.generationalMutationStrategy)
          .setMutationMagnitudeMode(config.mutationMagnitudeMode)
          .setFixedMutationValue(config.fixedMutationValue)
          .setEarlyStoppingScore(config.autoStoppingScore)
          .setEarlyStoppingFlag(config.autoStoppingFlag)
          .setEvolutionStrategy(config.evolutionStrategy)
          .setContinuousEvolutionMaxIterations(
            config.continuousEvolutionMaxIterations
          )
          .setContinuousEvolutionStoppingScore(
            config.continuousEvolutionStoppingScore
          )
          .setContinuousEvolutionParallelism(
            config.continuousEvolutionParallelism
          )
          .setContinuousEvolutionMutationAggressiveness(
            config.continuousEvolutionMutationAggressiveness
          )
          .setContinuousEvolutionGeneticMixing(
            config.continuousEvolutionGeneticMixing
          )
          .setContinuousEvolutionRollingImporvementCount(
            config.continuousEvolutionRollingImprovementCount
          )
          .setDataReductionFactor(config.dataReductionFactor)
          .setFirstGenMode(config.firstGenMode)
          .setFirstGenPermutations(config.firstGenPermutations)
          .setFirstGenIndexMixingMode(config.firstGenIndexMixingMode)
          .setFirstGenArraySeed(config.firstGenArraySeed)
          .evolveBest()
          .model
        modelType match {
          case Regressor =>
            val importances = rfModel
              .asInstanceOf[RandomForestRegressionModel]
              .featureImportances
              .toArray
            adjustedFieldNames.zip(importances).toMap[String, Double]

          case Classifier =>
            val importances = rfModel
              .asInstanceOf[RandomForestClassificationModel]
              .featureImportances
              .toArray
            adjustedFieldNames.zip(importances).toMap[String, Double]
        }
      case XGBoost =>
        val xgModel = new XGBoostTuner(df, config.modelType)
          .setLabelCol(config.labelCol)
          .setFeaturesCol(config.featuresCol)
          .setXGBoostNumericBoundaries(config.numericBoundaries)
          .setScoringMetric(config.scoringMetric)
          .setTrainPortion(config.trainPortion)
          .setTrainSplitMethod(config.trainSplitMethod)
          .setTrainSplitChronologicalColumn(
            config.trainSplitChronologicalColumn
          )
          .setTrainSplitChronologicalRandomPercentage(
            config.trainSplitChronlogicalRandomPercentage
          )
          .setParallelism(config.parallelism)
          .setKFold(config.kFold)
          .setSeed(config.seed)
          .setOptimizationStrategy(config.scoringOptimizationStrategy)
          .setFirstGenerationGenePool(config.firstGenerationGenePool)
          .setNumberOfMutationGenerations(config.numberOfGenerations)
          .setNumberOfMutationsPerGeneration(
            config.numberOfMutationsPerGeneration
          )
          .setNumberOfParentsToRetain(config.numberOfParentsToRetain)
          .setGeneticMixing(config.geneticMixing)
          .setGenerationalMutationStrategy(config.generationalMutationStrategy)
          .setMutationMagnitudeMode(config.mutationMagnitudeMode)
          .setFixedMutationValue(config.fixedMutationValue)
          .setEarlyStoppingScore(config.autoStoppingScore)
          .setEarlyStoppingFlag(config.autoStoppingFlag)
          .setEvolutionStrategy(config.evolutionStrategy)
          .setContinuousEvolutionMaxIterations(
            config.continuousEvolutionMaxIterations
          )
          .setContinuousEvolutionStoppingScore(
            config.continuousEvolutionStoppingScore
          )
          .setContinuousEvolutionParallelism(
            config.continuousEvolutionParallelism
          )
          .setContinuousEvolutionMutationAggressiveness(
            config.continuousEvolutionMutationAggressiveness
          )
          .setContinuousEvolutionGeneticMixing(
            config.continuousEvolutionGeneticMixing
          )
          .setContinuousEvolutionRollingImporvementCount(
            config.continuousEvolutionRollingImprovementCount
          )
          .setDataReductionFactor(config.dataReductionFactor)
          .setFirstGenMode(config.firstGenMode)
          .setFirstGenPermutations(config.firstGenPermutations)
          .setFirstGenIndexMixingMode(config.firstGenIndexMixingMode)
          .setFirstGenArraySeed(config.firstGenArraySeed)
          .evolveBest()
          .model
        modelType match {
          case Regressor =>
            xgModel
              .asInstanceOf[XGBoostRegressionModel]
              .nativeBooster
              .getFeatureScore(adjustedFieldNames)
              .map { case (k, v) => k -> v.toDouble }
              .toMap
          case Classifier =>
            xgModel
              .asInstanceOf[XGBoostClassificationModel]
              .nativeBooster
              .getFeatureScore(adjustedFieldNames)
              .map { case (k, v) => k -> v.toDouble }
              .toMap
        }
    }
    result
  }

  private def getTopFeaturesCount(featureDataFrame: DataFrame,
                                  featureCount: Int): Array[String] = {
    featureDataFrame
      .sort(col(importanceCol).desc)
      .limit(featureCount)
      .collect()
      .map(x => x(0).toString)
  }

  private def getTopFeaturesValue(featureDataFrame: DataFrame,
                                  importanceValue: Double): Array[String] = {
    featureDataFrame
      .filter(col(importanceCol) >= importanceValue)
      .sort(col(importanceCol).desc)
      .collect()
      .map(x => x(0).toString)
  }

  private def getAllImportances(featureDataFrame: DataFrame): Array[String] = {
    featureDataFrame
      .sort(col(importanceCol).desc)
      .collect()
      .map(x => x(0).toString)
  }

  /**
    * Main method for retrieving Feature Importances
    * @return FeatureImportanceReturn: DataFrame of Importances, sorted top fields in an Array[String], the raw data
    *         with vector applied, the fields included in the vector as Array[String], and all fields as Array[String]
    */
  def generateFeatureImportances(): FeatureImportanceReturn = {

    import spark.implicits._

    val cleanedData = fillNaValues()

    val vectorOutput = createFeatureVector(cleanedData)

    val interactionSwitch = if (config.featureInteractionFlag) {
      interactFeatures(vectorOutput)
    } else vectorOutput

    val importances =
      getImportances(interactionSwitch.data, interactionSwitch.fieldsInVector)

    val importancesDF = importances.toSeq
      .toDF(featureCol, importanceCol)
      .orderBy(col(importanceCol).desc)

    val importancesDFOutput = modelFamily match {
      case XGBoost =>
        importancesDF
          .withColumn(featureCol, split(col(featureCol), "_si$")(0))
      case _ =>
        importancesDF
          .withColumn(importanceCol, col(importanceCol) * 100.0)
          .withColumn(featureCol, split(col(featureCol), "_si$")(0))
    }

    val topFieldArray = cutOff match {
      case Count     => getTopFeaturesCount(importancesDF, cutoffValue.toInt)
      case Threshold => getTopFeaturesValue(importancesDF, cutoffValue)
      case None      => getAllImportances(importancesDF)
    }

    new FeatureImportanceReturn(
      importances = importancesDFOutput,
      topFields = topFieldArray
    ) {
      override def data: DataFrame = vectorOutput.data

      override def fieldsInVector: Array[String] = vectorOutput.fieldsInVector

      override def allFields: Array[String] = vectorOutput.allFields
    }
  }
}

object FeatureImportances {

  def apply(data: DataFrame,
            config: FeatureImportanceConfig,
            cutoffType: String,
            cutoffValue: Double): FeatureImportances =
    new FeatureImportances(data, config, cutoffType, cutoffValue)

}
