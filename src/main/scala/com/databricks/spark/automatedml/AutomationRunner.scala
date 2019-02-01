package com.databricks.spark.automatedml

import com.databricks.spark.automatedml.executor.DataPrep
import com.databricks.spark.automatedml.inference.{InferenceModelConfig, InferenceTools}
import com.databricks.spark.automatedml.model._
import com.databricks.spark.automatedml.params._
import com.databricks.spark.automatedml.reports.{DecisionTreeSplits, RandomForestFeatureImportance}
import com.databricks.spark.automatedml.tracking.MLFlowTracker
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, GBTRegressionModel, LinearRegressionModel, RandomForestRegressionModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer

class AutomationRunner(df: DataFrame) extends DataPrep(df) with InferenceTools {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private def runRandomForest(): (Array[RandomForestModelsWithResults], DataFrame, String, DataFrame) = {

    val (data, fields, modelSelection) = prepData()

    val cachedData = data.persist(StorageLevel.MEMORY_AND_DISK)
    cachedData.count

    val initialize = new RandomForestTuner(cachedData, modelSelection)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setRandomForestNumericBoundaries(_mainConfig.numericBoundaries)
      .setRandomForestStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(trainSplitValidation(_mainConfig.geneticConfig.trainSplitMethod, modelSelection))
      .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
      .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
      .setParallelism(_mainConfig.geneticConfig.parallelism)
      .setKFold(_mainConfig.geneticConfig.kFold)
      .setSeed(_mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
      .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
      .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
      .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
      .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
      .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
      .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
      .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
      .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
      .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
      .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
      .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
      .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
      .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)

    if(_modelSeedSetStatus) initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

    val (modelResults, modelStats) = initialize.evolveWithScoringDF()

    (modelResults, modelStats, modelSelection, cachedData)
  }

  private def runMLPC(): (Array[MLPCModelsWithResults], DataFrame, String, DataFrame) = {

    val (data, fields, modelSelection) = prepData()

    val cachedData = data.persist(StorageLevel.MEMORY_AND_DISK)
    cachedData.count

    modelSelection match {
      case "classifier" =>
        val initialize = new MLPCTuner(cachedData)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setMlpcNumericBoundaries(_mainConfig.numericBoundaries)
          .setMlpcStringBoundaries(_mainConfig.stringBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(trainSplitValidation(_mainConfig.geneticConfig.trainSplitMethod, modelSelection))
          .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
          .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
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
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
          .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
          .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
          .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
          .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
          .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)

        if(_modelSeedSetStatus) initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

        val (modelResults, modelStats) = initialize.evolveWithScoringDF()

        (modelResults, modelStats, modelSelection, cachedData)
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by MultiLayer Perceptron Classifier")
    }
  }

  private def runGBT(): (Array[GBTModelsWithResults], DataFrame, String, DataFrame) = {

    val (data, fields, modelSelection) = prepData()

    val cachedData = data.persist(StorageLevel.MEMORY_AND_DISK)
    cachedData.count

     val initialize = new GBTreesTuner(cachedData, modelSelection)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setGBTNumericBoundaries(_mainConfig.numericBoundaries)
      .setGBTStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(trainSplitValidation(_mainConfig.geneticConfig.trainSplitMethod, modelSelection))
      .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
      .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
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
      .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
      .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
      .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
      .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
      .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
      .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
      .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
      .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
      .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)

    if(_modelSeedSetStatus) initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

    val (modelResults, modelStats) = initialize.evolveWithScoringDF()

    (modelResults, modelStats, modelSelection, cachedData)
  }

  private def runLinearRegression(): (Array[LinearRegressionModelsWithResults], DataFrame, String, DataFrame) = {

    val (data, fields, modelSelection) = prepData()

    val cachedData = data.persist(StorageLevel.MEMORY_AND_DISK)
    cachedData.count

    modelSelection match {
      case "regressor" =>
        val initialize = new LinearRegressionTuner(cachedData)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setLinearRegressionNumericBoundaries(_mainConfig.numericBoundaries)
          .setLinearRegressionStringBoundaries(_mainConfig.stringBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(trainSplitValidation(_mainConfig.geneticConfig.trainSplitMethod, modelSelection))
          .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
          .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
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
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
          .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
          .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
          .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
          .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
          .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)

        if(_modelSeedSetStatus) initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

        val (modelResults, modelStats) = initialize.evolveWithScoringDF()

        (modelResults, modelStats, modelSelection, cachedData)
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by Linear Regression")
    }

  }

  private def runLogisticRegression(): (Array[LogisticRegressionModelsWithResults], DataFrame,  String, DataFrame) = {

    val (data, fields, modelSelection) = prepData()

    val cachedData = data.persist(StorageLevel.MEMORY_AND_DISK)
    cachedData.count

    modelSelection match {
      case "classifier" =>
        val initialize = new LogisticRegressionTuner(cachedData)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setLogisticRegressionNumericBoundaries(_mainConfig.numericBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(trainSplitValidation(_mainConfig.geneticConfig.trainSplitMethod, modelSelection))
          .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
          .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
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
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
          .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
          .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
          .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
          .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
          .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)

        if(_modelSeedSetStatus) initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

        val (modelResults, modelStats) = initialize.evolveWithScoringDF()

        (modelResults, modelStats, modelSelection, cachedData)
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by Logistic Regression")
    }

  }

  private def runSVM(): (Array[SVMModelsWithResults], DataFrame, String, DataFrame) = {

    val (data, fields, modelSelection) = prepData()

    val cachedData = data.persist(StorageLevel.MEMORY_AND_DISK)
    cachedData.count

    modelSelection match {
      case "classifier" =>
        val initialize = new SVMTuner(cachedData)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setSvmNumericBoundaries(_mainConfig.numericBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(trainSplitValidation(_mainConfig.geneticConfig.trainSplitMethod, modelSelection))
          .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
          .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
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
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
          .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
          .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
          .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
          .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
          .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)

        if(_modelSeedSetStatus) initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

        val (modelResults, modelStats) = initialize.evolveWithScoringDF()

        (modelResults, modelStats, modelSelection, cachedData)
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by Support Vector Machines")
    }
  }

 private def runTrees(): (Array[TreesModelsWithResults], DataFrame, String, DataFrame) = {

   val (data, fields, modelSelection) = prepData()

   val cachedData = data.persist(StorageLevel.MEMORY_AND_DISK)
   cachedData.count

   val initialize = new DecisionTreeTuner(data, modelSelection)
     .setLabelCol(_mainConfig.labelCol)
     .setFeaturesCol(_mainConfig.featuresCol)
     .setTreesNumericBoundaries(_mainConfig.numericBoundaries)
     .setTreesStringBoundaries(_mainConfig.stringBoundaries)
     .setScoringMetric(_mainConfig.scoringMetric)
     .setScoringMetric(_mainConfig.scoringMetric)
     .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
     .setTrainSplitMethod(trainSplitValidation(_mainConfig.geneticConfig.trainSplitMethod, modelSelection))
     .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
     .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
     .setParallelism(_mainConfig.geneticConfig.parallelism)
     .setKFold(_mainConfig.geneticConfig.kFold)
     .setSeed(_mainConfig.geneticConfig.seed)
     .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
     .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
     .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
     .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
     .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
     .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
     .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
     .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
     .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
     .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
     .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
     .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
     .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
     .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
     .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
     .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
     .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
     .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)

   if(_modelSeedSetStatus) initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

   val (modelResults, modelStats) = initialize.evolveWithScoringDF()

   (modelResults, modelStats, modelSelection, cachedData)
 }

  private def logResultsToMlFlow(runData: Array[GenericModelReturn], modelFamily: String, modelType: String): String = {

    val mlFlowLogger = new MLFlowTracker()
      .setMlFlowTrackingURI(_mainConfig.mlFlowConfig.mlFlowTrackingURI)
      .setMlFlowHostedAPIToken(_mainConfig.mlFlowConfig.mlFlowAPIToken)
      .setMlFlowExperimentName(_mainConfig.mlFlowConfig.mlFlowExperimentName)
      .setModelSaveDirectory(_mainConfig.mlFlowConfig.mlFlowModelSaveDirectory)

    if(_mainConfig.mlFlowLogArtifactsFlag) mlFlowLogger.logArtifactsOn() else mlFlowLogger.logArtifactsOff()

    try {
      mlFlowLogger.logMlFlowDataAndModels(runData, modelFamily, modelType, _mainConfig.inferenceConfigSaveLocation)
      "Logged to MlFlow Successful"
    } catch {
      case e: Exception =>
        val stack = e.toString
        val topStackTrace : String = e.getStackTrace.mkString("\n")
        println(s"Failed to log to mlflow. Check configuration. \n  $stack \n Top trace: \t $topStackTrace")
        logger.log(Level.INFO, stack)
        "Failed to Log to MlFlow"
    }

  }


  private def executeTuning(): TunerOutput = {

    val genericResults = new ArrayBuffer[GenericModelReturn]

    val (resultArray, modelStats, modelSelection, dataframe) = _mainConfig.modelFamily match {
      case "RandomForest" =>
        val (results, stats, selection, data) = runRandomForest()
        results.foreach{ x=>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection, data)
      case "GBT" =>
        val (results, stats, selection, data) = runGBT()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection, data)
      case "MLPC" =>
        val (results, stats, selection, data) = runMLPC()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection, data)
      case "LinearRegression" =>
        val (results, stats, selection, data) = runLinearRegression()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection, data)
      case "LogisticRegression" =>
        val (results, stats, selection, data) = runLogisticRegression()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection, data)
      case "SVM" =>
        val (results, stats, selection, data) = runSVM()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection, data)
      case "Trees" =>
        val (results, stats, selection, data) = runTrees()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection, data)
    }

    val genericResultData = genericResults.result.toArray

    if(_mainConfig.mlFlowLoggingFlag) {
      val mlFlowResult = logResultsToMlFlow(genericResultData, _mainConfig.modelFamily, modelSelection)
      println(mlFlowResult)
      logger.log(Level.INFO, mlFlowResult)
    } else {

      // set the Inference details in general for the run
      val inferenceModelConfig = InferenceModelConfig(
        modelFamily = _mainConfig.modelFamily,
        modelType = modelSelection,
        modelLoadMethod = "path",
        mlFlowConfig = _mainConfig.mlFlowConfig,
        mlFlowRunId = "none",
        modelPathLocation = _mainConfig.mlFlowConfig.mlFlowModelSaveDirectory
      )

      // Set the Inference Config
      setInferenceModelConfig(inferenceModelConfig)
      setInferenceConfigStorageLocation(_mainConfig.inferenceConfigSaveLocation)

      // Write the Inference Payload out to the specified location
      val outputInferencePayload = getInferenceConfig

      val inferenceConfigReadable = convertInferenceConfigToJson(outputInferencePayload)
      val inferenceLog = s"Inference Configuration: \n${inferenceConfigReadable.prettyJson}"
      println(inferenceLog)

      logger.log(Level.INFO, inferenceLog)

      if (_mainConfig.mlFlowConfig.mlFlowModelSaveDirectory.nonEmpty) {
        val inferenceConfigAsDF = convertInferenceConfigToDataFrame(outputInferencePayload)

        inferenceConfigAsDF.write.save(_mainConfig.inferenceConfigSaveLocation)
      }
    }

    val generationalData = extractGenerationalScores(genericResultData, _mainConfig.scoringOptimizationStrategy,
      _mainConfig.modelFamily, modelSelection)

    new TunerOutput(rawData = dataframe, modelSelection = modelSelection){
      override def modelReport: Array[GenericModelReturn] = genericResultData
      override def generationReport: Array[GenerationalReport] = generationalData
      override def modelReportDataFrame: DataFrame = modelStats
      override def generationReportDataFrame: DataFrame =
        generationDataFrameReport(generationalData, _mainConfig.scoringOptimizationStrategy)
    }

  }

  private def predictFromBestModel(resultPayload: Array[GenericModelReturn], rawData: DataFrame,
                                   modelSelection: String): DataFrame = {

    val bestModel = resultPayload(0)

    _mainConfig.modelFamily match {
      case "RandomForest" =>
        modelSelection match {
          case "regressor" =>
            val model = bestModel.model.asInstanceOf[RandomForestRegressionModel]
            model.transform(rawData)
          case "classifier" =>
            val model = bestModel.model.asInstanceOf[RandomForestClassificationModel]
            model.transform(rawData)
        }
      case "GBT" =>
        modelSelection match {
          case "regressor" =>
            val model = bestModel.model.asInstanceOf[GBTRegressionModel]
            model.transform(rawData)
          case "classifier" =>
            val model = bestModel.model.asInstanceOf[GBTClassificationModel]
            model.transform(rawData)
        }
      case "MLPC" =>
        val model = bestModel.model.asInstanceOf[MultilayerPerceptronClassificationModel]
        model.transform(rawData)
      case "LinearRegression" =>
        val model = bestModel.model.asInstanceOf[LinearRegressionModel]
        model.transform(rawData)
      case "LogisticRegression" =>
        val model = bestModel.model.asInstanceOf[LogisticRegressionModel]
        model.transform(rawData)
      case "SVM" =>
        val model = bestModel.model.asInstanceOf[LinearSVCModel]
        model.transform(rawData)
      case "Trees" =>
        modelSelection match {
          case "regressor" =>
            val model = bestModel.model.asInstanceOf[DecisionTreeClassificationModel]
            model.transform(rawData)
          case "classifier" =>
            val model = bestModel.model.asInstanceOf[DecisionTreeRegressionModel]
            model.transform(rawData)
        }
    }

  }

  private def exploreFeatureImportances(): (RandomForestModelsWithResults, DataFrame, Array[String]) = {

    val (data, fields, modelType) = prepData()

    val cachedData = data.persist(StorageLevel.MEMORY_AND_DISK)
    cachedData.count

    val featureResults = new RandomForestFeatureImportance(cachedData, _featureImportancesConfig, modelType)
      .setCutoffType(_mainConfig.featureImportanceCutoffType)
      .setCutoffValue(_mainConfig.featureImportanceCutoffValue)
      .runFeatureImportances(fields)
    cachedData.unpersist()

    featureResults
  }

  def runWithFeatureCulling(): FeatureImportanceOutput = {

    // Get the Feature Importances

    val (modelResults, importanceDF, culledFields) = exploreFeatureImportances()

    val selectableFields = culledFields :+ _mainConfig.labelCol

    val dataSubset = df.select(selectableFields.map(col):_*).persist(StorageLevel.MEMORY_AND_DISK)
    dataSubset.count
    
    val runResults = new AutomationRunner(dataSubset).setMainConfig(_mainConfig).run()

    dataSubset.unpersist()

    new FeatureImportanceOutput(importanceDF) {
      override def modelReport: Array[GenericModelReturn] = runResults.modelReport
      override def generationReport: Array[GenerationalReport] = runResults.generationReport
      override def modelReportDataFrame: DataFrame = runResults.modelReportDataFrame
      override def generationReportDataFrame: DataFrame = runResults.generationReportDataFrame
    }

  }

  def runFeatureCullingWithPrediction(): FeatureImportancePredictionOutput = {

    val (modelResults, importanceDF, culledFields) = exploreFeatureImportances()

    val selectableFields = culledFields :+ _mainConfig.labelCol

    val dataSubset = df.select(selectableFields.map(col):_*).persist(StorageLevel.MEMORY_AND_DISK)
    dataSubset.count

    val runResults = new AutomationRunner(dataSubset).setMainConfig(_mainConfig).executeTuning()

    dataSubset.unpersist()

    val predictedData = predictFromBestModel(runResults.modelReport, runResults.rawData, runResults.modelSelection)

    runResults.rawData.unpersist()

    new FeatureImportancePredictionOutput(
      featureImportances = importanceDF,
      predictionData = predictedData
    ) {
      override def modelReport: Array[GenericModelReturn] = runResults.modelReport
      override def generationReport: Array[GenerationalReport] = runResults.generationReport
      override def modelReportDataFrame: DataFrame = runResults.modelReportDataFrame
      override def generationReportDataFrame: DataFrame = runResults.generationReportDataFrame
    }

  }

  def generateDecisionSplits(): TreeSplitReport = {

    val (data, fields, modelType) = prepData()

    new DecisionTreeSplits(data, _treeSplitsConfig, modelType).runTreeSplitAnalysis(fields)

  }

  def run(): AutomationOutput = {

    val tunerResult = executeTuning()

    new AutomationOutput {
      override def modelReport: Array[GenericModelReturn] = tunerResult.modelReport
      override def generationReport: Array[GenerationalReport] = tunerResult.generationReport
      override def modelReportDataFrame: DataFrame = tunerResult.modelReportDataFrame
      override def generationReportDataFrame: DataFrame = tunerResult.generationReportDataFrame
    }
//
//    AutomationOutput(
//      modelReport = tunerResult.modelReport,
//      generationReport = tunerResult.generationReport,
//      modelReportDataFrame = tunerResult.modelReportDataFrame,
//      generationReportDataFrame = tunerResult.generationReportDataFrame
//    )

  }

  def runWithPrediction(): PredictionOutput = {

    val tunerResult = executeTuning()

    val predictedData = predictFromBestModel(tunerResult.modelReport, tunerResult.rawData, tunerResult.modelSelection)

    tunerResult.rawData.unpersist()

    new PredictionOutput(dataWithPredictions = predictedData) {
      override def modelReport: Array[GenericModelReturn] = tunerResult.modelReport
      override def generationReport: Array[GenerationalReport] = tunerResult.generationReport
      override def modelReportDataFrame: DataFrame = tunerResult.modelReportDataFrame
      override def generationReportDataFrame: DataFrame = tunerResult.generationReportDataFrame
    }

  }

 def runWithConfusionReport(): ConfusionOutput = {
   val predictionPayload = runWithPrediction()

   val confusionData = predictionPayload.dataWithPredictions
     .select("prediction", _labelCol)
     .groupBy("prediction", _labelCol)
     .agg(count("*").alias("count"))

   new ConfusionOutput(
     predictionData = predictionPayload.dataWithPredictions,
     confusionData = confusionData
   ) {
     override def modelReport: Array[GenericModelReturn] = predictionPayload.modelReport
     override def generationReport: Array[GenerationalReport] = predictionPayload.generationReport
     override def modelReportDataFrame: DataFrame = predictionPayload.modelReportDataFrame
     override def generationReportDataFrame: DataFrame = predictionPayload.generationReportDataFrame
   }

 }



  //TODO: add a generational runner to find the best model in a modelType (classification / regression)
  //TODO: this will require a new configuration methodology (generationalRunnerConfig) that has all of the families
  //TODO: default configs within it. with setters to override individual parts.  Might want to make it its own class.
}
