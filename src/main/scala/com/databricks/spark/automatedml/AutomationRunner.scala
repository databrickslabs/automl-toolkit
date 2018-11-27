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

  //TODO: this needs to override the numeric and string configs for Regression vs Classification!!!!




  private def runRandomForest(df: DataFrame): (Array[RandomForestModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = dataPrep(df)

    val (modelResults, modelStats) = new RandomForestTuner(data, modelSelection)
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

    (modelResults, modelStats, modelSelection)

  }

  private def runMLPC(df: DataFrame): (Array[MLPCModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = dataPrep(df)

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

    (modelResults, modelStats, modelSelection)
  }


  private def runGBT(df: DataFrame): (Array[GBTModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = dataPrep(df)

    val (modelResults, modelStats) = new GBTreesTuner(data, modelSelection)
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

    (modelResults, modelStats, modelSelection)
  }

  //TODO: Generation median, average, and stddev for scoring. (Help with full model search decision making)

  def extractFeatureImportances(df: DataFrame): (RandomForestModelsWithResults, DataFrame) = {

    val (data, fields, modelType) = dataPrep(df)

    new RandomForestFeatureImportance(data, fields, _featureImportanceConfig).runFeatureImportances(modelType)

  }

  def run(data: DataFrame): (Array[GenericModelReturn], Array[GenerationalReport], DataFrame, DataFrame) = {

    val genericResults = new ArrayBuffer[GenericModelReturn]

    val (resultArray, modelStats, modelSelection) = _mainConfig.modelType match {
      case "RandomForest" =>
        val (results, stats, selection) = runRandomForest(data)
        results.foreach{ x=>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection)
      case "GBT" =>
        val (results, stats, selection) = runGBT(data)
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection)
      case "MLPC" =>
        val (results, stats, selection) = runMLPC(data)
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection)
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
    //TODO: generation report
    val genericResultData = genericResults.result.toArray
    val generationalData = extractGenerationalScores(genericResultData, _mainConfig.scoringOptimizationStrategy,
      _mainConfig.modelType, modelSelection)

  (genericResults.result.toArray, generationalData, modelStats, generationDataFrameReport(generationalData,
    _mainConfig.scoringOptimizationStrategy))
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