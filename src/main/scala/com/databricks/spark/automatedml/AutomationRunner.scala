package com.databricks.spark.automatedml

import com.databricks.spark.automatedml.executor.Automation
import com.databricks.spark.automatedml.model.{GBTreesTuner, MLPCTuner, RandomForestTuner}
import com.databricks.spark.automatedml.params._
import com.databricks.spark.automatedml.reports.RandomForestFeatureImportance
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class AutomationRunner() extends Automation {

  //_mainConfig = getMainConfig //TODO: this probably isn't needed.

  //TODO: validation checks for type of model selected and the model's capabilities (i.e. don't try to use a classifier
  //if the model type doesn't support it)

  private var _featureImportanceConfig = _featureImportancesDefaults
  //TODO: this needs to override the numeric and string configs for Regression vs Classification!!!!
  def setFeatConfig(value: MainConfig): this.type = {
    _featureImportanceConfig = value
    this
  }

  def getFeatConfig: MainConfig = _featureImportanceConfig

  private def runRandomForest(df: DataFrame): (Array[RandomForestModelsWithResults], DataFrame) = {

    val (data, fields, modelType) = dataPrep(df)

    val (modelResults, modelStats) = new RandomForestTuner(data, modelType)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setRandomForestNumericBoundaries(_mainConfig.numericBoundaries)
      .setRandomForestStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setKFold(_mainConfig.geneticConfig.kFold)
      .setSeed(_mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
      .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
      .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
      .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
      .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
      .evolveWithScoringDF()

    (modelResults, modelStats)

  }

  private def runMLPC(df: DataFrame): (Array[MLPCModelsWithResults], DataFrame) = {

    val (data, fields, modelType) = dataPrep(df)

    val (modelResults, modelStats) = new MLPCTuner(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setMlpcNumericBoundaries(_mainConfig.numericBoundaries)
      .setMlpcStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setKFold(_mainConfig.geneticConfig.kFold)
      .setSeed(_mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
      .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
      .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
      .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
      .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
      .evolveWithScoringDF()

    (modelResults, modelStats)
  }


  private def runGBT(df: DataFrame): (Array[GBTModelsWithResults], DataFrame) = {

    val (data, fields, modelType) = dataPrep(df)

    val (modelResults, modelStats) = new GBTreesTuner(data, modelType)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setGBTNumericBoundaries(_mainConfig.numericBoundaries)
      .setGBTStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setKFold(_mainConfig.geneticConfig.kFold)
      .setSeed(_mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
      .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
      .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
      .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
      .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
      .evolveWithScoringDF()

    (modelResults, modelStats)
  }


  def extractFeatureImportances(data: DataFrame): (RandomForestModelsWithResults, DataFrame) = {

    val (data, fields, modelType) = dataPrep(data)

    new RandomForestFeatureImportance(data, fields, _featureImportancesDefaults).runFeatureImportances()

  }

  def run(data: DataFrame): (Array[GenericModelReturn], DataFrame) = {

    val genericResults = new ArrayBuffer[GenericModelReturn]

    val (resultArray, modelStats) = _mainConfig.modelType match {
      case "RandomForest" => {
        val (results, stats) = runRandomForest(data)
        results.foreach{ x=>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats)

      }
      case "GBT" =>
        val (results, stats) = runGBT(data)
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats)
      case "MLPC" =>
        val (results, stats) = runMLPC(data)
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats)
    }

//    resultArray.foreach{ x=>
//      genericResults += GenericModelReturn(
//        hyperParams = extractPayload(x.modelHyperParams),
//        model = x.model,
//        score = x.score,
//        metrics = x.evalMetrics,
//        generation = x.generation
//      )
//    }
  (genericResults.toArray, modelStats)
  }

}


//object AutomationRunner {
//
//}

/**
  * Import Config (which elements to do) and their settings
  * Run pipeline
  * Extract Fields
  * Filter / Sanitize
  * Run chosen Model
  * Extract Best
  * Run Report for Feature Importances
  * Run Report for Decision Tree
  * Export Reports + Importances + Models + Final DataFrame
  */