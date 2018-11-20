package com.databricks.spark.automatedml

import com.databricks.spark.automatedml.executor.Automation
import com.databricks.spark.automatedml.model.{MLPCTuner, RandomForestTuner}
import com.databricks.spark.automatedml.params.{MLPCModelsWithResults, MainConfig, RandomForestModelsWithResults}
import org.apache.spark.sql.DataFrame


class AutomationRunner(conf: MainConfig) extends Automation(conf){

  def run(): (Array[RandomForestModelsWithResults], DataFrame) = {

    val (data, fields, modelType) = dataPrep()

    new RandomForestTuner(data, modelType)
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

    //TODO: get reporting stats of the run out here.
    //TODO: Starting Fields, Filtered Fields, Filtered Data, Which elements ran in data preprocessing
  }

  def run()(implicit di: DummyImplicit): (Array[MLPCModelsWithResults], DataFrame) = {

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