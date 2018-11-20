package com.databricks.spark.automatedml

import com.databricks.spark.automatedml.executor.Automation
import com.databricks.spark.automatedml.model.{GBTreesTuner, MLPCTuner, RandomForestTuner}
import com.databricks.spark.automatedml.params.{GBTModelsWithResults, MLPCModelsWithResults, MainConfig, RandomForestModelsWithResults}
import org.apache.spark.sql.DataFrame


class AutomationRunner(conf: MainConfig) extends Automation(conf){

  def runRandomForest(): (Array[RandomForestModelsWithResults], DataFrame) = {

    val (data, fields, modelType) = dataPrep()

    val (modelResults, modelStats) = new RandomForestTuner(data, modelType)
      .setLabelCol(conf.labelCol)
      .setFeaturesCol(conf.featuresCol)
      .setRandomForestNumericBoundaries(_modelParams.numericBoundaries)
      .setRandomForestStringBoundaries(_modelParams.stringBoundaries)
      .setScoringMetric(_modelParams.scoringMetric)
      .setTrainPortion(_geneticConfig.trainPortion)
      .setKFold(_geneticConfig.kFold)
      .setSeed(_geneticConfig.seed)
      .setOptimizationStrategy(_scoringOptimizationStrategy)
      .setFirstGenerationGenePool(_geneticConfig.firstGenerationGenePool)
      .setNumberOfMutationsPerGeneration(_geneticConfig.numberOfMutationsPerGeneration)
      .setNumberOfParentsToRetain(_geneticConfig.numberOfParentsToRetain)
      .setNumberOfMutationsPerGeneration(_geneticConfig.numberOfMutationsPerGeneration)
      .setGeneticMixing(_geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(_geneticConfig.generationalMutationStrategy)
      .setMutationMagnitudeMode(_geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_geneticConfig.fixedMutationValue)
      .evolveWithScoringDF()

    (modelResults, modelStats)

    //TODO: get reporting stats of the run out here.
    //TODO: Starting Fields, Filtered Fields, Filtered Data, Which elements ran in data preprocessing
    //TODO: extract the hyperparameter payload into a Map object so that it can be generically stored
  }

  def runMLPC(): (Array[MLPCModelsWithResults], DataFrame) = {

    val (data, fields, modelType) = dataPrep()

    new MLPCTuner(data)
      .setLabelCol(conf.labelCol)
      .setFeaturesCol(conf.featuresCol)
      .setMlpcNumericBoundaries(_modelParams.numericBoundaries)
      .setMlpcStringBoundaries(_modelParams.stringBoundaries)
      .setScoringMetric(_modelParams.scoringMetric)
      .setTrainPortion(_geneticConfig.trainPortion)
      .setKFold(_geneticConfig.kFold)
      .setSeed(_geneticConfig.seed)
      .setOptimizationStrategy(_scoringOptimizationStrategy)
      .setFirstGenerationGenePool(_geneticConfig.firstGenerationGenePool)
      .setNumberOfMutationsPerGeneration(_geneticConfig.numberOfMutationsPerGeneration)
      .setNumberOfParentsToRetain(_geneticConfig.numberOfParentsToRetain)
      .setNumberOfMutationsPerGeneration(_geneticConfig.numberOfMutationsPerGeneration)
      .setGeneticMixing(_geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(_geneticConfig.generationalMutationStrategy)
      .setMutationMagnitudeMode(_geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_geneticConfig.fixedMutationValue)
      .evolveWithScoringDF()
  }


  def runGBT(): (Array[GBTModelsWithResults], DataFrame) = {

    val (data, fields, modelType) = dataPrep()

    new GBTreesTuner(data, modelType)
      .setLabelCol(conf.labelCol)
      .setFeaturesCol(conf.featuresCol)
      .setRGBTNumericBoundaries(_modelParams.numericBoundaries)
      .setGBTStringBoundaries(_modelParams.stringBoundaries)
      .setScoringMetric(_modelParams.scoringMetric)
      .setTrainPortion(_geneticConfig.trainPortion)
      .setKFold(_geneticConfig.kFold)
      .setSeed(_geneticConfig.seed)
      .setOptimizationStrategy(_scoringOptimizationStrategy)
      .setFirstGenerationGenePool(_geneticConfig.firstGenerationGenePool)
      .setNumberOfMutationsPerGeneration(_geneticConfig.numberOfMutationsPerGeneration)
      .setNumberOfParentsToRetain(_geneticConfig.numberOfParentsToRetain)
      .setNumberOfMutationsPerGeneration(_geneticConfig.numberOfMutationsPerGeneration)
      .setGeneticMixing(_geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(_geneticConfig.generationalMutationStrategy)
      .setMutationMagnitudeMode(_geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_geneticConfig.fixedMutationValue)
      .evolveWithScoringDF()
  }

}


object AutomationRunner {

}

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