package com.databricks.labs.automl.reports

import com.databricks.labs.automl.model.RandomForestTuner
import com.databricks.labs.automl.params.{
  MainConfig,
  RandomForestModelsWithResults
}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.DataFrame

class RandomForestFeatureImportance(data: DataFrame,
                                    featConfig: MainConfig,
                                    modelType: String)
    extends ReportingTools {

  final private val allowableCutoffTypes = List("none", "value", "count")

  private var _cutoffType = "count"

  private var _cutoffValue = 15.0

  def setCutoffType(value: String): this.type = {
    require(
      allowableCutoffTypes.contains(value),
      s"Cutoff type $value is not in ${allowableCutoffTypes.mkString(", ")}"
    )
    _cutoffType = value
    this
  }

  def setCutoffValue(value: Double): this.type = {
    _cutoffValue = value
    this
  }

  def getCutoffType: String = _cutoffType

  def getCutoffValue: Double = _cutoffValue

  def runFeatureImportances(
    fields: Array[String]
  ): (RandomForestModelsWithResults, DataFrame, Array[String]) = {

    val (modelResults, modelStats) = new RandomForestTuner(data, modelType)
      .setLabelCol(featConfig.labelCol)
      .setFeaturesCol(featConfig.featuresCol)
      .setRandomForestNumericBoundaries(featConfig.numericBoundaries)
      .setRandomForestStringBoundaries(featConfig.stringBoundaries)
      .setScoringMetric(featConfig.scoringMetric)
      .setTrainPortion(featConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(featConfig.geneticConfig.trainSplitMethod)
      .setTrainSplitChronologicalColumn(
        featConfig.geneticConfig.trainSplitChronologicalColumn
      )
      .setTrainSplitChronologicalRandomPercentage(
        featConfig.geneticConfig.trainSplitChronologicalRandomPercentage
      )
      .setParallelism(featConfig.geneticConfig.parallelism)
      .setKFold(featConfig.geneticConfig.kFold)
      .setSeed(featConfig.geneticConfig.seed)
      .setOptimizationStrategy(featConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(
        featConfig.geneticConfig.firstGenerationGenePool
      )
      .setNumberOfMutationGenerations(
        featConfig.geneticConfig.numberOfGenerations
      )
      .setNumberOfMutationsPerGeneration(
        featConfig.geneticConfig.numberOfMutationsPerGeneration
      )
      .setNumberOfParentsToRetain(
        featConfig.geneticConfig.numberOfParentsToRetain
      )
      .setGeneticMixing(featConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(
        featConfig.geneticConfig.generationalMutationStrategy
      )
      .setMutationMagnitudeMode(featConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(featConfig.geneticConfig.fixedMutationValue)
      .setEarlyStoppingScore(featConfig.autoStoppingScore)
      .setEarlyStoppingFlag(featConfig.autoStoppingFlag)
      .setEvolutionStrategy(featConfig.geneticConfig.evolutionStrategy)
      .setContinuousEvolutionMaxIterations(
        featConfig.geneticConfig.continuousEvolutionMaxIterations
      )
      .setContinuousEvolutionStoppingScore(
        featConfig.geneticConfig.continuousEvolutionStoppingScore
      )
      .setContinuousEvolutionParallelism(
        featConfig.geneticConfig.continuousEvolutionParallelism
      )
      .setContinuousEvolutionMutationAggressiveness(
        featConfig.geneticConfig.continuousEvolutionMutationAggressiveness
      )
      .setContinuousEvolutionGeneticMixing(
        featConfig.geneticConfig.continuousEvolutionGeneticMixing
      )
      .setContinuousEvolutionRollingImporvementCount(
        featConfig.geneticConfig.continuousEvolutionRollingImprovementCount
      )
      .evolveWithScoringDF()

    val bestModelData = modelResults.head
    val bestModelFeatureImportances = modelType match {
      case "classifier" =>
        bestModelData.model
          .asInstanceOf[RandomForestClassificationModel]
          .featureImportances
          .toArray
      case "regressor" =>
        bestModelData.model
          .asInstanceOf[RandomForestRegressionModel]
          .featureImportances
          .toArray
      case _ =>
        throw new UnsupportedOperationException(
          s"The model type provided, '${featConfig.modelFamily}', is not supported."
        )
    }

    val importances = generateFrameReport(fields, bestModelFeatureImportances)

    val extractedFields = _cutoffType match {
      case "none"  => fields
      case "value" => extractTopFeaturesByImportance(importances, _cutoffValue)
      case "count" => extractTopFeaturesByCount(importances, _cutoffValue.toInt)
      case _ =>
        throw new UnsupportedOperationException(
          s"Extraction mode ${_cutoffType} is not supported for feature importance reduction"
        )
    }

    (bestModelData, importances, extractedFields)

  }

}
