package com.databricks.spark.automatedml

import com.databricks.spark.automatedml.executor.Automation
import com.databricks.spark.automatedml.model._
import com.databricks.spark.automatedml.params._
import com.databricks.spark.automatedml.reports.{DecisionTreeSplits, RandomForestFeatureImportance}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class AutomationRunner(df: DataFrame) extends Automation {

  private def runRandomForest(): (Array[RandomForestModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = dataPrep(df)

    val (modelResults, modelStats) = new RandomForestTuner(data, modelSelection)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setRandomForestNumericBoundaries(_mainConfig.numericBoundaries)
      .setRandomForestStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setParallelism(_mainConfig.geneticConfig.parallelism)
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

  private def runMLPC(): (Array[MLPCModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = dataPrep(df)

    modelSelection match {
      case "classifier" =>
        val (modelResults, modelStats) = new MLPCTuner(data)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setMlpcNumericBoundaries(_mainConfig.numericBoundaries)
          .setMlpcStringBoundaries(_mainConfig.stringBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setParallelism(_mainConfig.geneticConfig.parallelism)
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
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by MultiLayer Perceptron Classifier")
    }
  }

  private def runGBT(): (Array[GBTModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = dataPrep(df)

    val (modelResults, modelStats) = new GBTreesTuner(data, modelSelection)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setGBTNumericBoundaries(_mainConfig.numericBoundaries)
      .setGBTStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setParallelism(_mainConfig.geneticConfig.parallelism)
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

  private def runLinearRegression(): (Array[LinearRegressionModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = dataPrep(df)

    modelSelection match {
      case "regressor" =>
        val (modelResults, modelStats) = new LinearRegressionTuner(data)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setLinearRegressionNumericBoundaries(_mainConfig.numericBoundaries)
          .setLinearRegressionStringBoundaries(_mainConfig.stringBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setParallelism(_mainConfig.geneticConfig.parallelism)
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
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by Linear Regression")
    }

  }

  private def runLogisticRegression(): (Array[LogisticRegressionModelsWithResults], DataFrame,  String) = {

    val (data, fields, modelSelection) = dataPrep(df)

    modelSelection match {
      case "classifier" =>
        val (modelResults, modelStats) = new LogisticRegressionTuner(data)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setLogisticRegressionNumericBoundaries(_mainConfig.numericBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setParallelism(_mainConfig.geneticConfig.parallelism)
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
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by Logistic Regression")
    }

  }

  private def runSVM(): (Array[SVMModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = dataPrep(df)

    modelSelection match {
      case "classifier" =>
        val (modelResults, modelStats) = new SVMTuner(data)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setSvmNumericBoundaries(_mainConfig.numericBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setParallelism(_mainConfig.geneticConfig.parallelism)
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
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by Support Vector Machines")
    }
  }

 private def runTrees(): (Array[TreesModelsWithResults], DataFrame, String) = {

   val (data, fields, modelSelection) = dataPrep(df)

   val (modelResults, modelStats) = new DecisionTreeTuner(data, modelSelection)
     .setLabelCol(_mainConfig.labelCol)
     .setFeaturesCol(_mainConfig.featuresCol)
     .setTreesNumericBoundaries(_mainConfig.numericBoundaries)
     .setTreesStringBoundaries(_mainConfig.stringBoundaries)
     .setScoringMetric(_mainConfig.scoringMetric)
     .setScoringMetric(_mainConfig.scoringMetric)
     .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
     .setParallelism(_mainConfig.geneticConfig.parallelism)
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

  //TODO: have all of the return types return a case class instance to allow for single variable returns and positional notation to extract the data!!

  def exploreFeatureImportances(): (RandomForestModelsWithResults, DataFrame) = {

    val (data, fields, modelType) = dataPrep(df)

    new RandomForestFeatureImportance(data, _featureImportancesConfig, modelType).runFeatureImportances(fields)

  }

  def generateDecisionSplits(): (String, DataFrame, Any) = {

    val (data, fields, modelType) = dataPrep(df)

    new DecisionTreeSplits(data, _treeSplitsConfig, modelType).runTreeSplitAnalysis(fields)

  }

  def run(): (Array[GenericModelReturn], Array[GenerationalReport], DataFrame, DataFrame) = {

    val genericResults = new ArrayBuffer[GenericModelReturn]

    val (resultArray, modelStats, modelSelection) = _mainConfig.modelFamily match {
      case "RandomForest" =>
        val (results, stats, selection) = runRandomForest()
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
        val (results, stats, selection) = runGBT()
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
        val (results, stats, selection) = runMLPC()
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
      case "LinearRegression" =>
        val (results, stats, selection) = runLinearRegression()
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
      case "LogisticRegression" =>
        val (results, stats, selection) = runLogisticRegression()
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
      case "SVM" =>
        val (results, stats, selection) = runSVM()
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
      case "Trees" =>
        val (results, stats, selection) = runTrees()
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

    val genericResultData = genericResults.result.toArray
    val generationalData = extractGenerationalScores(genericResultData, _mainConfig.scoringOptimizationStrategy,
      _mainConfig.modelFamily, modelSelection)

  (genericResults.result.toArray, generationalData, modelStats, generationDataFrameReport(generationalData,
    _mainConfig.scoringOptimizationStrategy))
  }

  //TODO: add a generational runner to find the best model in a modelType (classification / regression)
  //TODO: this will require a new configuration methodology (generationalRunnerConfig) that has all of the families
  //TODO: default configs within it. with setters to override individual parts.  Might want to make it its own class.
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