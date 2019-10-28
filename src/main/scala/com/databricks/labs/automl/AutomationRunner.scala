package com.databricks.labs.automl

import com.databricks.labs.automl.executor.DataPrep
import com.databricks.labs.automl.inference.{
  InferenceConfig,
  InferenceModelConfig,
  InferenceTools
}
import com.databricks.labs.automl.model._
import com.databricks.labs.automl.model.tools.PostModelingOptimization
import com.databricks.labs.automl.params._

import com.databricks.labs.automl.reports.{
  DecisionTreeSplits,
  RandomForestFeatureImportance
}
import com.databricks.labs.automl.tracking.{
  MLFlowReportStructure,
  MLFlowReturn,
  MLFlowTracker
}
import com.microsoft.ml.spark.lightgbm.{
  LightGBMClassificationModel,
  LightGBMRegressionModel
}
import ml.dmlc.xgboost4j.scala.spark.{
  XGBoostClassificationModel,
  XGBoostRegressionModel
}

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression.{
  DecisionTreeRegressionModel,
  GBTRegressionModel,
  LinearRegressionModel,
  RandomForestRegressionModel
}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.writePretty
import org.json4s.{Formats, NoTypeHints}

import scala.collection.mutable.ArrayBuffer

class AutomationRunner(df: DataFrame) extends DataPrep(df) with InferenceTools {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private def runRandomForest(
    payload: DataGeneration
  ): (Array[RandomForestModelsWithResults], DataFrame, String, DataFrame) = {

    val cachedData = if (_mainConfig.dataPrepCachingFlag) {
      payload.data.persist(StorageLevel.MEMORY_AND_DISK)
    } else {
      payload.data
    }

    if (_mainConfig.dataPrepCachingFlag) payload.data.count()

    val initialize = new RandomForestTuner(cachedData, payload.modelType)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setRandomForestNumericBoundaries(_mainConfig.numericBoundaries)
      .setRandomForestStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(
        trainSplitValidation(
          _mainConfig.geneticConfig.trainSplitMethod,
          payload.modelType
        )
      )
      .setSyntheticCol(_mainConfig.geneticConfig.kSampleConfig.syntheticCol)
      .setKGroups(_mainConfig.geneticConfig.kSampleConfig.kGroups)
      .setKMeansMaxIter(_mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter)
      .setKMeansTolerance(
        _mainConfig.geneticConfig.kSampleConfig.kMeansTolerance
      )
      .setKMeansDistanceMeasurement(
        _mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement
      )
      .setKMeansSeed(_mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
      .setKMeansPredictionCol(
        _mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol
      )
      .setLSHHashTables(_mainConfig.geneticConfig.kSampleConfig.lshHashTables)
      .setLSHSeed(_mainConfig.geneticConfig.kSampleConfig.lshSeed)
      .setLSHOutputCol(_mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
      .setQuorumCount(_mainConfig.geneticConfig.kSampleConfig.quorumCount)
      .setMinimumVectorCountToMutate(
        _mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate
      )
      .setVectorMutationMethod(
        _mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod
      )
      .setMutationMode(_mainConfig.geneticConfig.kSampleConfig.mutationMode)
      .setMutationValue(_mainConfig.geneticConfig.kSampleConfig.mutationValue)
      .setLabelBalanceMode(
        _mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
      )
      .setCardinalityThreshold(
        _mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
      )
      .setNumericRatio(_mainConfig.geneticConfig.kSampleConfig.numericRatio)
      .setNumericTarget(_mainConfig.geneticConfig.kSampleConfig.numericTarget)
      .setTrainSplitChronologicalColumn(
        _mainConfig.geneticConfig.trainSplitChronologicalColumn
      )
      .setTrainSplitChronologicalRandomPercentage(
        _mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage
      )
      .setParallelism(_mainConfig.geneticConfig.parallelism)
      .setKFold(_mainConfig.geneticConfig.kFold)
      .setSeed(_mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(
        _mainConfig.geneticConfig.firstGenerationGenePool
      )
      .setNumberOfMutationGenerations(
        _mainConfig.geneticConfig.numberOfGenerations
      )
      .setNumberOfMutationsPerGeneration(
        _mainConfig.geneticConfig.numberOfMutationsPerGeneration
      )
      .setNumberOfParentsToRetain(
        _mainConfig.geneticConfig.numberOfParentsToRetain
      )
      .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(
        _mainConfig.geneticConfig.generationalMutationStrategy
      )
      .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
      .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
      .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
      .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
      .setContinuousEvolutionImprovementThreshold(
        _mainConfig.geneticConfig.continuousEvolutionImprovementThreshold
      )
      .setGeneticMBORegressorType(
        _mainConfig.geneticConfig.geneticMBORegressorType
      )
      .setGeneticMBOCandidateFactor(
        _mainConfig.geneticConfig.geneticMBOCandidateFactor
      )
      .setContinuousEvolutionMaxIterations(
        _mainConfig.geneticConfig.continuousEvolutionMaxIterations
      )
      .setContinuousEvolutionStoppingScore(
        _mainConfig.geneticConfig.continuousEvolutionStoppingScore
      )
      .setContinuousEvolutionParallelism(
        _mainConfig.geneticConfig.continuousEvolutionParallelism
      )
      .setContinuousEvolutionMutationAggressiveness(
        _mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness
      )
      .setContinuousEvolutionGeneticMixing(
        _mainConfig.geneticConfig.continuousEvolutionGeneticMixing
      )
      .setContinuousEvolutionRollingImporvementCount(
        _mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount
      )
      .setDataReductionFactor(_mainConfig.dataReductionFactor)
      .setFirstGenMode(_mainConfig.geneticConfig.initialGenerationMode)
      .setFirstGenPermutations(
        _mainConfig.geneticConfig.initialGenerationConfig.permutationCount
      )
      .setFirstGenIndexMixingMode(
        _mainConfig.geneticConfig.initialGenerationConfig.indexMixingMode
      )
      .setFirstGenArraySeed(
        _mainConfig.geneticConfig.initialGenerationConfig.arraySeed
      )
      .setHyperSpaceModelCount(_mainConfig.geneticConfig.hyperSpaceModelCount)

    if (_modelSeedSetStatus)
      initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

    val (modelResultsRaw, modelStatsRaw) = initialize.evolveWithScoringDF()

    val resultBuffer = modelResultsRaw.toBuffer
    val statsBuffer = new ArrayBuffer[DataFrame]()
    statsBuffer += modelStatsRaw

    if (_mainConfig.geneticConfig.hyperSpaceInference) {

      println("\n\t\tStarting Post Tuning Inference Run.\n")

      val genericResults = new ArrayBuffer[GenericModelReturn]

      modelResultsRaw.foreach { x =>
        genericResults += GenericModelReturn(
          hyperParams = extractPayload(x.modelHyperParams),
          model = x.model,
          score = x.score,
          metrics = x.evalMetrics,
          generation = x.generation
        )
      }

      val hyperSpaceRunCandidates = new PostModelingOptimization()
        .setModelFamily("RandomForest")
        .setModelType(payload.modelType)
        .setHyperParameterSpaceCount(
          _mainConfig.geneticConfig.hyperSpaceInferenceCount
        )
        .setNumericBoundaries(_mainConfig.numericBoundaries)
        .setStringBoundaries(_mainConfig.stringBoundaries)
        .setSeed(_mainConfig.geneticConfig.seed)
        .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
        .randomForestPrediction(
          genericResults.result.toArray,
          _mainConfig.geneticConfig.hyperSpaceModelType,
          _mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =
        initialize.postRunModeledHyperParams(hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x
      }
      statsBuffer += hyperDataFrame

    }

    (
      resultBuffer.toArray,
      statsBuffer.reduce(_ union _),
      payload.modelType,
      cachedData
    )

  }

  private def runLightGBM(
    lightGBMType: String,
    payload: DataGeneration
  ): (Array[LightGBMModelsWithResults], DataFrame, String, DataFrame) = {

    val cachedData = if (_mainConfig.dataPrepCachingFlag) {
      payload.data.persist(StorageLevel.MEMORY_AND_DISK)
    } else {
      payload.data
    }

    if (_mainConfig.dataPrepCachingFlag) payload.data.count()

    val initialize = new LightGBMTuner(
      cachedData,
      payload.modelType,
      lightGBMType
    ).setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setLGBMNumericBoundaries(_mainConfig.numericBoundaries)
      .setLGBMStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(
        trainSplitValidation(
          _mainConfig.geneticConfig.trainSplitMethod,
          payload.modelType
        )
      )
      .setSyntheticCol(_mainConfig.geneticConfig.kSampleConfig.syntheticCol)
      .setKGroups(_mainConfig.geneticConfig.kSampleConfig.kGroups)
      .setKMeansMaxIter(_mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter)
      .setKMeansTolerance(
        _mainConfig.geneticConfig.kSampleConfig.kMeansTolerance
      )
      .setKMeansDistanceMeasurement(
        _mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement
      )
      .setKMeansSeed(_mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
      .setKMeansPredictionCol(
        _mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol
      )
      .setLSHHashTables(_mainConfig.geneticConfig.kSampleConfig.lshHashTables)
      .setLSHSeed(_mainConfig.geneticConfig.kSampleConfig.lshSeed)
      .setLSHOutputCol(_mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
      .setQuorumCount(_mainConfig.geneticConfig.kSampleConfig.quorumCount)
      .setMinimumVectorCountToMutate(
        _mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate
      )
      .setVectorMutationMethod(
        _mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod
      )
      .setMutationMode(_mainConfig.geneticConfig.kSampleConfig.mutationMode)
      .setMutationValue(_mainConfig.geneticConfig.kSampleConfig.mutationValue)
      .setLabelBalanceMode(
        _mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
      )
      .setCardinalityThreshold(
        _mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
      )
      .setNumericRatio(_mainConfig.geneticConfig.kSampleConfig.numericRatio)
      .setNumericTarget(_mainConfig.geneticConfig.kSampleConfig.numericTarget)
      .setTrainSplitChronologicalColumn(
        _mainConfig.geneticConfig.trainSplitChronologicalColumn
      )
      .setTrainSplitChronologicalRandomPercentage(
        _mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage
      )
      .setParallelism(_mainConfig.geneticConfig.parallelism)
      .setKFold(_mainConfig.geneticConfig.kFold)
      .setSeed(_mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(
        _mainConfig.geneticConfig.firstGenerationGenePool
      )
      .setNumberOfMutationGenerations(
        _mainConfig.geneticConfig.numberOfGenerations
      )
      .setNumberOfMutationsPerGeneration(
        _mainConfig.geneticConfig.numberOfMutationsPerGeneration
      )
      .setNumberOfParentsToRetain(
        _mainConfig.geneticConfig.numberOfParentsToRetain
      )
      .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(
        _mainConfig.geneticConfig.generationalMutationStrategy
      )
      .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
      .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
      .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
      .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
      .setContinuousEvolutionImprovementThreshold(
        _mainConfig.geneticConfig.continuousEvolutionImprovementThreshold
      )
      .setGeneticMBORegressorType(
        _mainConfig.geneticConfig.geneticMBORegressorType
      )
      .setGeneticMBOCandidateFactor(
        _mainConfig.geneticConfig.geneticMBOCandidateFactor
      )
      .setContinuousEvolutionMaxIterations(
        _mainConfig.geneticConfig.continuousEvolutionMaxIterations
      )
      .setContinuousEvolutionStoppingScore(
        _mainConfig.geneticConfig.continuousEvolutionStoppingScore
      )
      .setContinuousEvolutionParallelism(
        _mainConfig.geneticConfig.continuousEvolutionParallelism
      )
      .setContinuousEvolutionMutationAggressiveness(
        _mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness
      )
      .setContinuousEvolutionGeneticMixing(
        _mainConfig.geneticConfig.continuousEvolutionGeneticMixing
      )
      .setContinuousEvolutionRollingImporvementCount(
        _mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount
      )
      .setDataReductionFactor(_mainConfig.dataReductionFactor)
      .setFirstGenMode(_mainConfig.geneticConfig.initialGenerationMode)
      .setFirstGenPermutations(
        _mainConfig.geneticConfig.initialGenerationConfig.permutationCount
      )
      .setFirstGenIndexMixingMode(
        _mainConfig.geneticConfig.initialGenerationConfig.indexMixingMode
      )
      .setFirstGenArraySeed(
        _mainConfig.geneticConfig.initialGenerationConfig.arraySeed
      )
      .setHyperSpaceModelCount(_mainConfig.geneticConfig.hyperSpaceModelCount)

    if (_modelSeedSetStatus)
      initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

    val (modelResultsRaw, modelStatsRaw) = initialize.evolveWithScoringDF()

    val resultBuffer = modelResultsRaw.toBuffer
    val statsBuffer = new ArrayBuffer[DataFrame]()
    statsBuffer += modelStatsRaw

    if (_mainConfig.geneticConfig.hyperSpaceInference) {

      println("\n\t\tStarting Post Tuning Inference Run.\n")

      val genericResults = new ArrayBuffer[GenericModelReturn]

      modelResultsRaw.foreach { x =>
        genericResults += GenericModelReturn(
          hyperParams = extractPayload(x.modelHyperParams),
          model = x.model,
          score = x.score,
          metrics = x.evalMetrics,
          generation = x.generation
        )
      }

      val hyperSpaceRunCandidates = new PostModelingOptimization()
        .setModelFamily(lightGBMType)
        .setModelType(payload.modelType)
        .setHyperParameterSpaceCount(
          _mainConfig.geneticConfig.hyperSpaceInferenceCount
        )
        .setNumericBoundaries(_mainConfig.numericBoundaries)
        .setStringBoundaries(_mainConfig.stringBoundaries)
        .setSeed(_mainConfig.geneticConfig.seed)
        .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
        .lightGBMPrediction(
          genericResults.result.toArray,
          _mainConfig.geneticConfig.hyperSpaceModelType,
          _mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =
        initialize.postRunModeledHyperParams(hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x
      }
      statsBuffer += hyperDataFrame

    }

    (
      resultBuffer.toArray,
      statsBuffer.reduce(_ union _),
      payload.modelType,
      cachedData
    )

  }

  private def runXGBoost(
    payload: DataGeneration
  ): (Array[XGBoostModelsWithResults], DataFrame, String, DataFrame) = {

    val cachedData = if (_mainConfig.dataPrepCachingFlag) {
      payload.data.persist(StorageLevel.MEMORY_AND_DISK)
    } else {
      payload.data
    }

    if (_mainConfig.dataPrepCachingFlag) payload.data.count()

    val initialize = new XGBoostTuner(cachedData, payload.modelType)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setXGBoostNumericBoundaries(_mainConfig.numericBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(
        trainSplitValidation(
          _mainConfig.geneticConfig.trainSplitMethod,
          payload.modelType
        )
      )
      .setSyntheticCol(_mainConfig.geneticConfig.kSampleConfig.syntheticCol)
      .setKGroups(_mainConfig.geneticConfig.kSampleConfig.kGroups)
      .setKMeansMaxIter(_mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter)
      .setKMeansTolerance(
        _mainConfig.geneticConfig.kSampleConfig.kMeansTolerance
      )
      .setKMeansDistanceMeasurement(
        _mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement
      )
      .setKMeansSeed(_mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
      .setKMeansPredictionCol(
        _mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol
      )
      .setLSHHashTables(_mainConfig.geneticConfig.kSampleConfig.lshHashTables)
      .setLSHSeed(_mainConfig.geneticConfig.kSampleConfig.lshSeed)
      .setLSHOutputCol(_mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
      .setQuorumCount(_mainConfig.geneticConfig.kSampleConfig.quorumCount)
      .setMinimumVectorCountToMutate(
        _mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate
      )
      .setVectorMutationMethod(
        _mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod
      )
      .setMutationMode(_mainConfig.geneticConfig.kSampleConfig.mutationMode)
      .setMutationValue(_mainConfig.geneticConfig.kSampleConfig.mutationValue)
      .setLabelBalanceMode(
        _mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
      )
      .setCardinalityThreshold(
        _mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
      )
      .setNumericRatio(_mainConfig.geneticConfig.kSampleConfig.numericRatio)
      .setNumericTarget(_mainConfig.geneticConfig.kSampleConfig.numericTarget)
      .setTrainSplitChronologicalColumn(
        _mainConfig.geneticConfig.trainSplitChronologicalColumn
      )
      .setTrainSplitChronologicalRandomPercentage(
        _mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage
      )
      .setParallelism(_mainConfig.geneticConfig.parallelism)
      .setKFold(_mainConfig.geneticConfig.kFold)
      .setSeed(_mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(
        _mainConfig.geneticConfig.firstGenerationGenePool
      )
      .setNumberOfMutationGenerations(
        _mainConfig.geneticConfig.numberOfGenerations
      )
      .setNumberOfMutationsPerGeneration(
        _mainConfig.geneticConfig.numberOfMutationsPerGeneration
      )
      .setNumberOfParentsToRetain(
        _mainConfig.geneticConfig.numberOfParentsToRetain
      )
      .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(
        _mainConfig.geneticConfig.generationalMutationStrategy
      )
      .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
      .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
      .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
      .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
      .setContinuousEvolutionImprovementThreshold(
        _mainConfig.geneticConfig.continuousEvolutionImprovementThreshold
      )
      .setGeneticMBORegressorType(
        _mainConfig.geneticConfig.geneticMBORegressorType
      )
      .setGeneticMBOCandidateFactor(
        _mainConfig.geneticConfig.geneticMBOCandidateFactor
      )
      .setContinuousEvolutionMaxIterations(
        _mainConfig.geneticConfig.continuousEvolutionMaxIterations
      )
      .setContinuousEvolutionStoppingScore(
        _mainConfig.geneticConfig.continuousEvolutionStoppingScore
      )
      .setContinuousEvolutionParallelism(
        _mainConfig.geneticConfig.continuousEvolutionParallelism
      )
      .setContinuousEvolutionMutationAggressiveness(
        _mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness
      )
      .setContinuousEvolutionGeneticMixing(
        _mainConfig.geneticConfig.continuousEvolutionGeneticMixing
      )
      .setContinuousEvolutionRollingImporvementCount(
        _mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount
      )
      .setDataReductionFactor(_mainConfig.dataReductionFactor)
      .setFirstGenMode(_mainConfig.geneticConfig.initialGenerationMode)
      .setFirstGenPermutations(
        _mainConfig.geneticConfig.initialGenerationConfig.permutationCount
      )
      .setFirstGenIndexMixingMode(
        _mainConfig.geneticConfig.initialGenerationConfig.indexMixingMode
      )
      .setFirstGenArraySeed(
        _mainConfig.geneticConfig.initialGenerationConfig.arraySeed
      )
      .setHyperSpaceModelCount(_mainConfig.geneticConfig.hyperSpaceModelCount)

    if (_modelSeedSetStatus)
      initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

    val (modelResultsRaw, modelStatsRaw) = initialize.evolveWithScoringDF()

    val resultBuffer = modelResultsRaw.toBuffer
    val statsBuffer = new ArrayBuffer[DataFrame]()
    statsBuffer += modelStatsRaw

    if (_mainConfig.geneticConfig.hyperSpaceInference) {

      println("\n\t\tStarting Post Tuning Inference Run.\n")

      val genericResults = new ArrayBuffer[GenericModelReturn]

      modelResultsRaw.foreach { x =>
        genericResults += GenericModelReturn(
          hyperParams = extractPayload(x.modelHyperParams),
          model = x.model,
          score = x.score,
          metrics = x.evalMetrics,
          generation = x.generation
        )
      }

      val hyperSpaceRunCandidates = new PostModelingOptimization()
        .setModelFamily("XGBoost")
        .setModelType(payload.modelType)
        .setHyperParameterSpaceCount(
          _mainConfig.geneticConfig.hyperSpaceInferenceCount
        )
        .setNumericBoundaries(_mainConfig.numericBoundaries)
        .setStringBoundaries(_mainConfig.stringBoundaries)
        .setSeed(_mainConfig.geneticConfig.seed)
        .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
        .xgBoostPrediction(
          genericResults.result.toArray,
          _mainConfig.geneticConfig.hyperSpaceModelType,
          _mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =
        initialize.postRunModeledHyperParams(hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x
      }
      statsBuffer += hyperDataFrame

    }

    (
      resultBuffer.toArray,
      statsBuffer.reduce(_ union _),
      payload.modelType,
      cachedData
    )

  }

  private def runMLPC(
    payload: DataGeneration
  ): (Array[MLPCModelsWithResults], DataFrame, String, DataFrame) = {

    val cachedData = if (_mainConfig.dataPrepCachingFlag) {
      payload.data.persist(StorageLevel.MEMORY_AND_DISK)
    } else {
      payload.data
    }

    if (_mainConfig.dataPrepCachingFlag) payload.data.count()

    payload.modelType match {
      case "classifier" =>
        val initialize = new MLPCTuner(cachedData)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setMlpcNumericBoundaries(_mainConfig.numericBoundaries)
          .setMlpcStringBoundaries(_mainConfig.stringBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(
            trainSplitValidation(
              _mainConfig.geneticConfig.trainSplitMethod,
              payload.modelType
            )
          )
          .setSyntheticCol(_mainConfig.geneticConfig.kSampleConfig.syntheticCol)
          .setKGroups(_mainConfig.geneticConfig.kSampleConfig.kGroups)
          .setKMeansMaxIter(
            _mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter
          )
          .setKMeansTolerance(
            _mainConfig.geneticConfig.kSampleConfig.kMeansTolerance
          )
          .setKMeansDistanceMeasurement(
            _mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement
          )
          .setKMeansSeed(_mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
          .setKMeansPredictionCol(
            _mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol
          )
          .setLSHHashTables(
            _mainConfig.geneticConfig.kSampleConfig.lshHashTables
          )
          .setLSHSeed(_mainConfig.geneticConfig.kSampleConfig.lshSeed)
          .setLSHOutputCol(_mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
          .setQuorumCount(_mainConfig.geneticConfig.kSampleConfig.quorumCount)
          .setMinimumVectorCountToMutate(
            _mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate
          )
          .setVectorMutationMethod(
            _mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod
          )
          .setMutationMode(_mainConfig.geneticConfig.kSampleConfig.mutationMode)
          .setMutationValue(
            _mainConfig.geneticConfig.kSampleConfig.mutationValue
          )
          .setLabelBalanceMode(
            _mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
          )
          .setCardinalityThreshold(
            _mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
          )
          .setNumericRatio(_mainConfig.geneticConfig.kSampleConfig.numericRatio)
          .setNumericTarget(
            _mainConfig.geneticConfig.kSampleConfig.numericTarget
          )
          .setTrainSplitChronologicalColumn(
            _mainConfig.geneticConfig.trainSplitChronologicalColumn
          )
          .setTrainSplitChronologicalRandomPercentage(
            _mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage
          )
          .setParallelism(_mainConfig.geneticConfig.parallelism)
          .setKFold(_mainConfig.geneticConfig.kFold)
          .setSeed(_mainConfig.geneticConfig.seed)
          .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
          .setFirstGenerationGenePool(
            _mainConfig.geneticConfig.firstGenerationGenePool
          )
          .setNumberOfMutationGenerations(
            _mainConfig.geneticConfig.numberOfGenerations
          )
          .setNumberOfMutationsPerGeneration(
            _mainConfig.geneticConfig.numberOfMutationsPerGeneration
          )
          .setNumberOfParentsToRetain(
            _mainConfig.geneticConfig.numberOfParentsToRetain
          )
          .setNumberOfMutationsPerGeneration(
            _mainConfig.geneticConfig.numberOfMutationsPerGeneration
          )
          .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
          .setGenerationalMutationStrategy(
            _mainConfig.geneticConfig.generationalMutationStrategy
          )
          .setMutationMagnitudeMode(
            _mainConfig.geneticConfig.mutationMagnitudeMode
          )
          .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionImprovementThreshold(
            _mainConfig.geneticConfig.continuousEvolutionImprovementThreshold
          )
          .setGeneticMBORegressorType(
            _mainConfig.geneticConfig.geneticMBORegressorType
          )
          .setGeneticMBOCandidateFactor(
            _mainConfig.geneticConfig.geneticMBOCandidateFactor
          )
          .setContinuousEvolutionMaxIterations(
            _mainConfig.geneticConfig.continuousEvolutionMaxIterations
          )
          .setContinuousEvolutionStoppingScore(
            _mainConfig.geneticConfig.continuousEvolutionStoppingScore
          )
          .setContinuousEvolutionParallelism(
            _mainConfig.geneticConfig.continuousEvolutionParallelism
          )
          .setContinuousEvolutionMutationAggressiveness(
            _mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness
          )
          .setContinuousEvolutionGeneticMixing(
            _mainConfig.geneticConfig.continuousEvolutionGeneticMixing
          )
          .setContinuousEvolutionRollingImporvementCount(
            _mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount
          )
          .setDataReductionFactor(_mainConfig.dataReductionFactor)
          .setFirstGenMode(_mainConfig.geneticConfig.initialGenerationMode)
          .setFirstGenPermutations(
            _mainConfig.geneticConfig.initialGenerationConfig.permutationCount
          )
          .setFirstGenIndexMixingMode(
            _mainConfig.geneticConfig.initialGenerationConfig.indexMixingMode
          )
          .setFirstGenArraySeed(
            _mainConfig.geneticConfig.initialGenerationConfig.arraySeed
          )
          .setHyperSpaceModelCount(
            _mainConfig.geneticConfig.hyperSpaceModelCount
          )

        if (_modelSeedSetStatus)
          initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

        val (modelResultsRaw, modelStatsRaw) = initialize.evolveWithScoringDF()

        val resultBuffer = modelResultsRaw.toBuffer
        val statsBuffer = new ArrayBuffer[DataFrame]()
        statsBuffer += modelStatsRaw

        if (_mainConfig.geneticConfig.hyperSpaceInference) {

          println("\n\t\tStarting Post Tuning Inference Run.\n")

          val genericResults = new ArrayBuffer[GenericModelReturn]

          modelResultsRaw.foreach { x =>
            genericResults += GenericModelReturn(
              hyperParams = extractPayload(x.modelHyperParams),
              model = x.model,
              score = x.score,
              metrics = x.evalMetrics,
              generation = x.generation
            )
          }

          val hyperSpaceRunCandidates = new PostModelingOptimization()
            .setModelFamily("MLPC")
            .setModelType(payload.modelType)
            .setHyperParameterSpaceCount(
              _mainConfig.geneticConfig.hyperSpaceInferenceCount
            )
            .setNumericBoundaries(_mainConfig.numericBoundaries)
            .setStringBoundaries(_mainConfig.stringBoundaries)
            .setSeed(_mainConfig.geneticConfig.seed)
            .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
            .mlpcPrediction(
              genericResults.result.toArray,
              _mainConfig.geneticConfig.hyperSpaceModelType,
              _mainConfig.geneticConfig.hyperSpaceModelCount,
              initialize.getFeatureInputSize,
              initialize.getClassDistinctCount
            )

          val (hyperResults, hyperDataFrame) =
            initialize.postRunModeledHyperParams(hyperSpaceRunCandidates)

          hyperResults.foreach { x =>
            resultBuffer += x
          }
          statsBuffer += hyperDataFrame

        }

        (
          resultBuffer.toArray,
          statsBuffer.reduce(_ union _),
          payload.modelType,
          cachedData
        )

      case _ =>
        throw new UnsupportedOperationException(
          s"Detected Model Type ${payload.modelType} is not supported by MultiLayer Perceptron Classifier"
        )
    }
  }

  private def runGBT(
    payload: DataGeneration
  ): (Array[GBTModelsWithResults], DataFrame, String, DataFrame) = {

    val cachedData = if (_mainConfig.dataPrepCachingFlag) {
      payload.data.persist(StorageLevel.MEMORY_AND_DISK)
    } else {
      payload.data
    }

    if (_mainConfig.dataPrepCachingFlag) payload.data.count()

    val initialize = new GBTreesTuner(cachedData, payload.modelType)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setGBTNumericBoundaries(_mainConfig.numericBoundaries)
      .setGBTStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(
        trainSplitValidation(
          _mainConfig.geneticConfig.trainSplitMethod,
          payload.modelType
        )
      )
      .setSyntheticCol(_mainConfig.geneticConfig.kSampleConfig.syntheticCol)
      .setKGroups(_mainConfig.geneticConfig.kSampleConfig.kGroups)
      .setKMeansMaxIter(_mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter)
      .setKMeansTolerance(
        _mainConfig.geneticConfig.kSampleConfig.kMeansTolerance
      )
      .setKMeansDistanceMeasurement(
        _mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement
      )
      .setKMeansSeed(_mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
      .setKMeansPredictionCol(
        _mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol
      )
      .setLSHHashTables(_mainConfig.geneticConfig.kSampleConfig.lshHashTables)
      .setLSHSeed(_mainConfig.geneticConfig.kSampleConfig.lshSeed)
      .setLSHOutputCol(_mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
      .setQuorumCount(_mainConfig.geneticConfig.kSampleConfig.quorumCount)
      .setMinimumVectorCountToMutate(
        _mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate
      )
      .setVectorMutationMethod(
        _mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod
      )
      .setMutationMode(_mainConfig.geneticConfig.kSampleConfig.mutationMode)
      .setMutationValue(_mainConfig.geneticConfig.kSampleConfig.mutationValue)
      .setLabelBalanceMode(
        _mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
      )
      .setCardinalityThreshold(
        _mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
      )
      .setNumericRatio(_mainConfig.geneticConfig.kSampleConfig.numericRatio)
      .setNumericTarget(_mainConfig.geneticConfig.kSampleConfig.numericTarget)
      .setTrainSplitChronologicalColumn(
        _mainConfig.geneticConfig.trainSplitChronologicalColumn
      )
      .setTrainSplitChronologicalRandomPercentage(
        _mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage
      )
      .setParallelism(_mainConfig.geneticConfig.parallelism)
      .setKFold(_mainConfig.geneticConfig.kFold)
      .setSeed(_mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(
        _mainConfig.geneticConfig.firstGenerationGenePool
      )
      .setNumberOfMutationGenerations(
        _mainConfig.geneticConfig.numberOfGenerations
      )
      .setNumberOfMutationsPerGeneration(
        _mainConfig.geneticConfig.numberOfMutationsPerGeneration
      )
      .setNumberOfParentsToRetain(
        _mainConfig.geneticConfig.numberOfParentsToRetain
      )
      .setNumberOfMutationsPerGeneration(
        _mainConfig.geneticConfig.numberOfMutationsPerGeneration
      )
      .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(
        _mainConfig.geneticConfig.generationalMutationStrategy
      )
      .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
      .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
      .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
      .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
      .setContinuousEvolutionImprovementThreshold(
        _mainConfig.geneticConfig.continuousEvolutionImprovementThreshold
      )
      .setGeneticMBORegressorType(
        _mainConfig.geneticConfig.geneticMBORegressorType
      )
      .setGeneticMBOCandidateFactor(
        _mainConfig.geneticConfig.geneticMBOCandidateFactor
      )
      .setContinuousEvolutionMaxIterations(
        _mainConfig.geneticConfig.continuousEvolutionMaxIterations
      )
      .setContinuousEvolutionStoppingScore(
        _mainConfig.geneticConfig.continuousEvolutionStoppingScore
      )
      .setContinuousEvolutionParallelism(
        _mainConfig.geneticConfig.continuousEvolutionParallelism
      )
      .setContinuousEvolutionMutationAggressiveness(
        _mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness
      )
      .setContinuousEvolutionGeneticMixing(
        _mainConfig.geneticConfig.continuousEvolutionGeneticMixing
      )
      .setContinuousEvolutionRollingImporvementCount(
        _mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount
      )
      .setDataReductionFactor(_mainConfig.dataReductionFactor)
      .setFirstGenMode(_mainConfig.geneticConfig.initialGenerationMode)
      .setFirstGenPermutations(
        _mainConfig.geneticConfig.initialGenerationConfig.permutationCount
      )
      .setFirstGenIndexMixingMode(
        _mainConfig.geneticConfig.initialGenerationConfig.indexMixingMode
      )
      .setFirstGenArraySeed(
        _mainConfig.geneticConfig.initialGenerationConfig.arraySeed
      )
      .setHyperSpaceModelCount(_mainConfig.geneticConfig.hyperSpaceModelCount)

    if (_modelSeedSetStatus)
      initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

    val (modelResultsRaw, modelStatsRaw) = initialize.evolveWithScoringDF()

    val resultBuffer = modelResultsRaw.toBuffer
    val statsBuffer = new ArrayBuffer[DataFrame]()
    statsBuffer += modelStatsRaw

    if (_mainConfig.geneticConfig.hyperSpaceInference) {

      println("\n\t\tStarting Post Tuning Inference Run.\n")

      val genericResults = new ArrayBuffer[GenericModelReturn]

      modelResultsRaw.foreach { x =>
        genericResults += GenericModelReturn(
          hyperParams = extractPayload(x.modelHyperParams),
          model = x.model,
          score = x.score,
          metrics = x.evalMetrics,
          generation = x.generation
        )
      }

      val hyperSpaceRunCandidates = new PostModelingOptimization()
        .setModelFamily("GBT")
        .setModelType(payload.modelType)
        .setHyperParameterSpaceCount(
          _mainConfig.geneticConfig.hyperSpaceInferenceCount
        )
        .setNumericBoundaries(_mainConfig.numericBoundaries)
        .setStringBoundaries(_mainConfig.stringBoundaries)
        .setSeed(_mainConfig.geneticConfig.seed)
        .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
        .gbtPrediction(
          genericResults.result.toArray,
          _mainConfig.geneticConfig.hyperSpaceModelType,
          _mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =
        initialize.postRunModeledHyperParams(hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x
      }
      statsBuffer += hyperDataFrame

    }

    (
      resultBuffer.toArray,
      statsBuffer.reduce(_ union _),
      payload.modelType,
      cachedData
    )

  }

  private def runLinearRegression(
    payload: DataGeneration
  ): (Array[LinearRegressionModelsWithResults], DataFrame, String, DataFrame) = {

    val cachedData = if (_mainConfig.dataPrepCachingFlag) {
      payload.data.persist(StorageLevel.MEMORY_AND_DISK)
    } else {
      payload.data
    }

    if (_mainConfig.dataPrepCachingFlag) payload.data.count()

    payload.modelType match {
      case "regressor" =>
        val initialize = new LinearRegressionTuner(cachedData)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setLinearRegressionNumericBoundaries(_mainConfig.numericBoundaries)
          .setLinearRegressionStringBoundaries(_mainConfig.stringBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(
            trainSplitValidation(
              _mainConfig.geneticConfig.trainSplitMethod,
              payload.modelType
            )
          )
          .setSyntheticCol(_mainConfig.geneticConfig.kSampleConfig.syntheticCol)
          .setKGroups(_mainConfig.geneticConfig.kSampleConfig.kGroups)
          .setKMeansMaxIter(
            _mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter
          )
          .setKMeansTolerance(
            _mainConfig.geneticConfig.kSampleConfig.kMeansTolerance
          )
          .setKMeansDistanceMeasurement(
            _mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement
          )
          .setKMeansSeed(_mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
          .setKMeansPredictionCol(
            _mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol
          )
          .setLSHHashTables(
            _mainConfig.geneticConfig.kSampleConfig.lshHashTables
          )
          .setLSHSeed(_mainConfig.geneticConfig.kSampleConfig.lshSeed)
          .setLSHOutputCol(_mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
          .setQuorumCount(_mainConfig.geneticConfig.kSampleConfig.quorumCount)
          .setMinimumVectorCountToMutate(
            _mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate
          )
          .setVectorMutationMethod(
            _mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod
          )
          .setMutationMode(_mainConfig.geneticConfig.kSampleConfig.mutationMode)
          .setMutationValue(
            _mainConfig.geneticConfig.kSampleConfig.mutationValue
          )
          .setLabelBalanceMode(
            _mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
          )
          .setCardinalityThreshold(
            _mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
          )
          .setNumericRatio(_mainConfig.geneticConfig.kSampleConfig.numericRatio)
          .setNumericTarget(
            _mainConfig.geneticConfig.kSampleConfig.numericTarget
          )
          .setTrainSplitChronologicalColumn(
            _mainConfig.geneticConfig.trainSplitChronologicalColumn
          )
          .setTrainSplitChronologicalRandomPercentage(
            _mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage
          )
          .setParallelism(_mainConfig.geneticConfig.parallelism)
          .setKFold(_mainConfig.geneticConfig.kFold)
          .setSeed(_mainConfig.geneticConfig.seed)
          .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
          .setFirstGenerationGenePool(
            _mainConfig.geneticConfig.firstGenerationGenePool
          )
          .setNumberOfMutationGenerations(
            _mainConfig.geneticConfig.numberOfGenerations
          )
          .setNumberOfMutationsPerGeneration(
            _mainConfig.geneticConfig.numberOfMutationsPerGeneration
          )
          .setNumberOfParentsToRetain(
            _mainConfig.geneticConfig.numberOfParentsToRetain
          )
          .setNumberOfMutationsPerGeneration(
            _mainConfig.geneticConfig.numberOfMutationsPerGeneration
          )
          .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
          .setGenerationalMutationStrategy(
            _mainConfig.geneticConfig.generationalMutationStrategy
          )
          .setMutationMagnitudeMode(
            _mainConfig.geneticConfig.mutationMagnitudeMode
          )
          .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionImprovementThreshold(
            _mainConfig.geneticConfig.continuousEvolutionImprovementThreshold
          )
          .setGeneticMBORegressorType(
            _mainConfig.geneticConfig.geneticMBORegressorType
          )
          .setGeneticMBOCandidateFactor(
            _mainConfig.geneticConfig.geneticMBOCandidateFactor
          )
          .setContinuousEvolutionMaxIterations(
            _mainConfig.geneticConfig.continuousEvolutionMaxIterations
          )
          .setContinuousEvolutionStoppingScore(
            _mainConfig.geneticConfig.continuousEvolutionStoppingScore
          )
          .setContinuousEvolutionParallelism(
            _mainConfig.geneticConfig.continuousEvolutionParallelism
          )
          .setContinuousEvolutionMutationAggressiveness(
            _mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness
          )
          .setContinuousEvolutionGeneticMixing(
            _mainConfig.geneticConfig.continuousEvolutionGeneticMixing
          )
          .setContinuousEvolutionRollingImporvementCount(
            _mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount
          )
          .setDataReductionFactor(_mainConfig.dataReductionFactor)
          .setFirstGenMode(_mainConfig.geneticConfig.initialGenerationMode)
          .setFirstGenPermutations(
            _mainConfig.geneticConfig.initialGenerationConfig.permutationCount
          )
          .setFirstGenIndexMixingMode(
            _mainConfig.geneticConfig.initialGenerationConfig.indexMixingMode
          )
          .setFirstGenArraySeed(
            _mainConfig.geneticConfig.initialGenerationConfig.arraySeed
          )
          .setHyperSpaceModelCount(
            _mainConfig.geneticConfig.hyperSpaceModelCount
          )

        if (_modelSeedSetStatus)
          initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

        val (modelResultsRaw, modelStatsRaw) = initialize.evolveWithScoringDF()

        val resultBuffer = modelResultsRaw.toBuffer
        val statsBuffer = new ArrayBuffer[DataFrame]()
        statsBuffer += modelStatsRaw

        if (_mainConfig.geneticConfig.hyperSpaceInference) {

          println("\n\t\tStarting Post Tuning Inference Run.\n")

          val genericResults = new ArrayBuffer[GenericModelReturn]

          modelResultsRaw.foreach { x =>
            genericResults += GenericModelReturn(
              hyperParams = extractPayload(x.modelHyperParams),
              model = x.model,
              score = x.score,
              metrics = x.evalMetrics,
              generation = x.generation
            )
          }

          val hyperSpaceRunCandidates = new PostModelingOptimization()
            .setModelFamily("LinearRegression")
            .setModelType(payload.modelType)
            .setHyperParameterSpaceCount(
              _mainConfig.geneticConfig.hyperSpaceInferenceCount
            )
            .setNumericBoundaries(_mainConfig.numericBoundaries)
            .setStringBoundaries(_mainConfig.stringBoundaries)
            .setSeed(_mainConfig.geneticConfig.seed)
            .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
            .linearRegressionPrediction(
              genericResults.result.toArray,
              _mainConfig.geneticConfig.hyperSpaceModelType,
              _mainConfig.geneticConfig.hyperSpaceModelCount
            )

          val (hyperResults, hyperDataFrame) =
            initialize.postRunModeledHyperParams(hyperSpaceRunCandidates)

          hyperResults.foreach { x =>
            resultBuffer += x
          }
          statsBuffer += hyperDataFrame

        }

        (
          resultBuffer.toArray,
          statsBuffer.reduce(_ union _),
          payload.modelType,
          cachedData
        )

      case _ =>
        throw new UnsupportedOperationException(
          s"Detected Model Type ${payload.modelType} is not supported by Linear Regression"
        )
    }

  }

  private def runLogisticRegression(payload: DataGeneration): (Array[
                                                                 LogisticRegressionModelsWithResults
                                                               ],
                                                               DataFrame,
                                                               String,
                                                               DataFrame) = {

    val cachedData = if (_mainConfig.dataPrepCachingFlag) {
      payload.data.persist(StorageLevel.MEMORY_AND_DISK)
    } else {
      payload.data
    }

    if (_mainConfig.dataPrepCachingFlag) payload.data.count()

    payload.modelType match {
      case "classifier" =>
        val initialize = new LogisticRegressionTuner(cachedData)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setLogisticRegressionNumericBoundaries(_mainConfig.numericBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(
            trainSplitValidation(
              _mainConfig.geneticConfig.trainSplitMethod,
              payload.modelType
            )
          )
          .setSyntheticCol(_mainConfig.geneticConfig.kSampleConfig.syntheticCol)
          .setKGroups(_mainConfig.geneticConfig.kSampleConfig.kGroups)
          .setKMeansMaxIter(
            _mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter
          )
          .setKMeansTolerance(
            _mainConfig.geneticConfig.kSampleConfig.kMeansTolerance
          )
          .setKMeansDistanceMeasurement(
            _mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement
          )
          .setKMeansSeed(_mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
          .setKMeansPredictionCol(
            _mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol
          )
          .setLSHHashTables(
            _mainConfig.geneticConfig.kSampleConfig.lshHashTables
          )
          .setLSHSeed(_mainConfig.geneticConfig.kSampleConfig.lshSeed)
          .setLSHOutputCol(_mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
          .setQuorumCount(_mainConfig.geneticConfig.kSampleConfig.quorumCount)
          .setMinimumVectorCountToMutate(
            _mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate
          )
          .setVectorMutationMethod(
            _mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod
          )
          .setMutationMode(_mainConfig.geneticConfig.kSampleConfig.mutationMode)
          .setMutationValue(
            _mainConfig.geneticConfig.kSampleConfig.mutationValue
          )
          .setLabelBalanceMode(
            _mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
          )
          .setCardinalityThreshold(
            _mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
          )
          .setNumericRatio(_mainConfig.geneticConfig.kSampleConfig.numericRatio)
          .setNumericTarget(
            _mainConfig.geneticConfig.kSampleConfig.numericTarget
          )
          .setTrainSplitChronologicalColumn(
            _mainConfig.geneticConfig.trainSplitChronologicalColumn
          )
          .setTrainSplitChronologicalRandomPercentage(
            _mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage
          )
          .setParallelism(_mainConfig.geneticConfig.parallelism)
          .setKFold(_mainConfig.geneticConfig.kFold)
          .setSeed(_mainConfig.geneticConfig.seed)
          .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
          .setFirstGenerationGenePool(
            _mainConfig.geneticConfig.firstGenerationGenePool
          )
          .setNumberOfMutationGenerations(
            _mainConfig.geneticConfig.numberOfGenerations
          )
          .setNumberOfMutationsPerGeneration(
            _mainConfig.geneticConfig.numberOfMutationsPerGeneration
          )
          .setNumberOfParentsToRetain(
            _mainConfig.geneticConfig.numberOfParentsToRetain
          )
          .setNumberOfMutationsPerGeneration(
            _mainConfig.geneticConfig.numberOfMutationsPerGeneration
          )
          .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
          .setGenerationalMutationStrategy(
            _mainConfig.geneticConfig.generationalMutationStrategy
          )
          .setMutationMagnitudeMode(
            _mainConfig.geneticConfig.mutationMagnitudeMode
          )
          .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionImprovementThreshold(
            _mainConfig.geneticConfig.continuousEvolutionImprovementThreshold
          )
          .setGeneticMBORegressorType(
            _mainConfig.geneticConfig.geneticMBORegressorType
          )
          .setGeneticMBOCandidateFactor(
            _mainConfig.geneticConfig.geneticMBOCandidateFactor
          )
          .setContinuousEvolutionMaxIterations(
            _mainConfig.geneticConfig.continuousEvolutionMaxIterations
          )
          .setContinuousEvolutionStoppingScore(
            _mainConfig.geneticConfig.continuousEvolutionStoppingScore
          )
          .setContinuousEvolutionParallelism(
            _mainConfig.geneticConfig.continuousEvolutionParallelism
          )
          .setContinuousEvolutionMutationAggressiveness(
            _mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness
          )
          .setContinuousEvolutionGeneticMixing(
            _mainConfig.geneticConfig.continuousEvolutionGeneticMixing
          )
          .setContinuousEvolutionRollingImporvementCount(
            _mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount
          )
          .setDataReductionFactor(_mainConfig.dataReductionFactor)
          .setFirstGenMode(_mainConfig.geneticConfig.initialGenerationMode)
          .setFirstGenPermutations(
            _mainConfig.geneticConfig.initialGenerationConfig.permutationCount
          )
          .setFirstGenIndexMixingMode(
            _mainConfig.geneticConfig.initialGenerationConfig.indexMixingMode
          )
          .setFirstGenArraySeed(
            _mainConfig.geneticConfig.initialGenerationConfig.arraySeed
          )
          .setHyperSpaceModelCount(
            _mainConfig.geneticConfig.hyperSpaceModelCount
          )

        if (_modelSeedSetStatus)
          initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

        val (modelResultsRaw, modelStatsRaw) = initialize.evolveWithScoringDF()

        val resultBuffer = modelResultsRaw.toBuffer
        val statsBuffer = new ArrayBuffer[DataFrame]()
        statsBuffer += modelStatsRaw

        if (_mainConfig.geneticConfig.hyperSpaceInference) {

          println("\n\t\tStarting Post Tuning Inference Run.\n")

          val genericResults = new ArrayBuffer[GenericModelReturn]

          modelResultsRaw.foreach { x =>
            genericResults += GenericModelReturn(
              hyperParams = extractPayload(x.modelHyperParams),
              model = x.model,
              score = x.score,
              metrics = x.evalMetrics,
              generation = x.generation
            )
          }

          val hyperSpaceRunCandidates = new PostModelingOptimization()
            .setModelFamily("LogisticRegression")
            .setModelType(payload.modelType)
            .setHyperParameterSpaceCount(
              _mainConfig.geneticConfig.hyperSpaceInferenceCount
            )
            .setNumericBoundaries(_mainConfig.numericBoundaries)
            .setStringBoundaries(_mainConfig.stringBoundaries)
            .setSeed(_mainConfig.geneticConfig.seed)
            .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
            .logisticRegressionPrediction(
              genericResults.result.toArray,
              _mainConfig.geneticConfig.hyperSpaceModelType,
              _mainConfig.geneticConfig.hyperSpaceModelCount
            )

          val (hyperResults, hyperDataFrame) =
            initialize.postRunModeledHyperParams(hyperSpaceRunCandidates)

          hyperResults.foreach { x =>
            resultBuffer += x
          }
          statsBuffer += hyperDataFrame

        }

        (
          resultBuffer.toArray,
          statsBuffer.reduce(_ union _),
          payload.modelType,
          cachedData
        )

      case _ =>
        throw new UnsupportedOperationException(
          s"Detected Model Type ${payload.modelType} is not supported by Logistic Regression"
        )
    }

  }

  private def runSVM(
    payload: DataGeneration
  ): (Array[SVMModelsWithResults], DataFrame, String, DataFrame) = {

    val cachedData = if (_mainConfig.dataPrepCachingFlag) {
      payload.data.persist(StorageLevel.MEMORY_AND_DISK)
    } else {
      payload.data
    }

    if (_mainConfig.dataPrepCachingFlag) payload.data.count()

    payload.modelType match {
      case "classifier" =>
        val initialize = new SVMTuner(cachedData)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setSvmNumericBoundaries(_mainConfig.numericBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(
            trainSplitValidation(
              _mainConfig.geneticConfig.trainSplitMethod,
              payload.modelType
            )
          )
          .setSyntheticCol(_mainConfig.geneticConfig.kSampleConfig.syntheticCol)
          .setKGroups(_mainConfig.geneticConfig.kSampleConfig.kGroups)
          .setKMeansMaxIter(
            _mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter
          )
          .setKMeansTolerance(
            _mainConfig.geneticConfig.kSampleConfig.kMeansTolerance
          )
          .setKMeansDistanceMeasurement(
            _mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement
          )
          .setKMeansSeed(_mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
          .setKMeansPredictionCol(
            _mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol
          )
          .setLSHHashTables(
            _mainConfig.geneticConfig.kSampleConfig.lshHashTables
          )
          .setLSHSeed(_mainConfig.geneticConfig.kSampleConfig.lshSeed)
          .setLSHOutputCol(_mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
          .setQuorumCount(_mainConfig.geneticConfig.kSampleConfig.quorumCount)
          .setMinimumVectorCountToMutate(
            _mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate
          )
          .setVectorMutationMethod(
            _mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod
          )
          .setMutationMode(_mainConfig.geneticConfig.kSampleConfig.mutationMode)
          .setMutationValue(
            _mainConfig.geneticConfig.kSampleConfig.mutationValue
          )
          .setLabelBalanceMode(
            _mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
          )
          .setCardinalityThreshold(
            _mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
          )
          .setNumericRatio(_mainConfig.geneticConfig.kSampleConfig.numericRatio)
          .setNumericTarget(
            _mainConfig.geneticConfig.kSampleConfig.numericTarget
          )
          .setTrainSplitChronologicalColumn(
            _mainConfig.geneticConfig.trainSplitChronologicalColumn
          )
          .setTrainSplitChronologicalRandomPercentage(
            _mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage
          )
          .setParallelism(_mainConfig.geneticConfig.parallelism)
          .setKFold(_mainConfig.geneticConfig.kFold)
          .setSeed(_mainConfig.geneticConfig.seed)
          .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
          .setFirstGenerationGenePool(
            _mainConfig.geneticConfig.firstGenerationGenePool
          )
          .setNumberOfMutationGenerations(
            _mainConfig.geneticConfig.numberOfGenerations
          )
          .setNumberOfMutationsPerGeneration(
            _mainConfig.geneticConfig.numberOfMutationsPerGeneration
          )
          .setNumberOfParentsToRetain(
            _mainConfig.geneticConfig.numberOfParentsToRetain
          )
          .setNumberOfMutationsPerGeneration(
            _mainConfig.geneticConfig.numberOfMutationsPerGeneration
          )
          .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
          .setGenerationalMutationStrategy(
            _mainConfig.geneticConfig.generationalMutationStrategy
          )
          .setMutationMagnitudeMode(
            _mainConfig.geneticConfig.mutationMagnitudeMode
          )
          .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionImprovementThreshold(
            _mainConfig.geneticConfig.continuousEvolutionImprovementThreshold
          )
          .setGeneticMBORegressorType(
            _mainConfig.geneticConfig.geneticMBORegressorType
          )
          .setGeneticMBOCandidateFactor(
            _mainConfig.geneticConfig.geneticMBOCandidateFactor
          )
          .setContinuousEvolutionMaxIterations(
            _mainConfig.geneticConfig.continuousEvolutionMaxIterations
          )
          .setContinuousEvolutionStoppingScore(
            _mainConfig.geneticConfig.continuousEvolutionStoppingScore
          )
          .setContinuousEvolutionParallelism(
            _mainConfig.geneticConfig.continuousEvolutionParallelism
          )
          .setContinuousEvolutionMutationAggressiveness(
            _mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness
          )
          .setContinuousEvolutionGeneticMixing(
            _mainConfig.geneticConfig.continuousEvolutionGeneticMixing
          )
          .setContinuousEvolutionRollingImporvementCount(
            _mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount
          )
          .setDataReductionFactor(_mainConfig.dataReductionFactor)
          .setFirstGenMode(_mainConfig.geneticConfig.initialGenerationMode)
          .setFirstGenPermutations(
            _mainConfig.geneticConfig.initialGenerationConfig.permutationCount
          )
          .setFirstGenIndexMixingMode(
            _mainConfig.geneticConfig.initialGenerationConfig.indexMixingMode
          )
          .setFirstGenArraySeed(
            _mainConfig.geneticConfig.initialGenerationConfig.arraySeed
          )
          .setHyperSpaceModelCount(
            _mainConfig.geneticConfig.hyperSpaceModelCount
          )

        if (_modelSeedSetStatus)
          initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

        val (modelResultsRaw, modelStatsRaw) = initialize.evolveWithScoringDF()

        val resultBuffer = modelResultsRaw.toBuffer
        val statsBuffer = new ArrayBuffer[DataFrame]()
        statsBuffer += modelStatsRaw

        if (_mainConfig.geneticConfig.hyperSpaceInference) {

          println("\n\t\tStarting Post Tuning Inference Run.\n")

          val genericResults = new ArrayBuffer[GenericModelReturn]

          modelResultsRaw.foreach { x =>
            genericResults += GenericModelReturn(
              hyperParams = extractPayload(x.modelHyperParams),
              model = x.model,
              score = x.score,
              metrics = x.evalMetrics,
              generation = x.generation
            )
          }

          val hyperSpaceRunCandidates = new PostModelingOptimization()
            .setModelFamily("SVM")
            .setModelType(payload.modelType)
            .setHyperParameterSpaceCount(
              _mainConfig.geneticConfig.hyperSpaceInferenceCount
            )
            .setNumericBoundaries(_mainConfig.numericBoundaries)
            .setStringBoundaries(_mainConfig.stringBoundaries)
            .setSeed(_mainConfig.geneticConfig.seed)
            .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
            .svmPrediction(
              genericResults.result.toArray,
              _mainConfig.geneticConfig.hyperSpaceModelType,
              _mainConfig.geneticConfig.hyperSpaceModelCount
            )

          val (hyperResults, hyperDataFrame) =
            initialize.postRunModeledHyperParams(hyperSpaceRunCandidates)

          hyperResults.foreach { x =>
            resultBuffer += x
          }
          statsBuffer += hyperDataFrame

        }

        (
          resultBuffer.toArray,
          statsBuffer.reduce(_ union _),
          payload.modelType,
          cachedData
        )

      case _ =>
        throw new UnsupportedOperationException(
          s"Detected Model Type ${payload.modelType} is not supported by Support Vector Machines"
        )
    }
  }

  private def runTrees(
    payload: DataGeneration
  ): (Array[TreesModelsWithResults], DataFrame, String, DataFrame) = {

    val cachedData = if (_mainConfig.dataPrepCachingFlag) {
      payload.data.persist(StorageLevel.MEMORY_AND_DISK)
    } else {
      payload.data
    }

    if (_mainConfig.dataPrepCachingFlag) payload.data.count()

    val initialize = new DecisionTreeTuner(payload.data, payload.modelType)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setTreesNumericBoundaries(_mainConfig.numericBoundaries)
      .setTreesStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(
        trainSplitValidation(
          _mainConfig.geneticConfig.trainSplitMethod,
          payload.modelType
        )
      )
      .setSyntheticCol(_mainConfig.geneticConfig.kSampleConfig.syntheticCol)
      .setKGroups(_mainConfig.geneticConfig.kSampleConfig.kGroups)
      .setKMeansMaxIter(_mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter)
      .setKMeansTolerance(
        _mainConfig.geneticConfig.kSampleConfig.kMeansTolerance
      )
      .setKMeansDistanceMeasurement(
        _mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement
      )
      .setKMeansSeed(_mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
      .setKMeansPredictionCol(
        _mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol
      )
      .setLSHHashTables(_mainConfig.geneticConfig.kSampleConfig.lshHashTables)
      .setLSHSeed(_mainConfig.geneticConfig.kSampleConfig.lshSeed)
      .setLSHOutputCol(_mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
      .setQuorumCount(_mainConfig.geneticConfig.kSampleConfig.quorumCount)
      .setMinimumVectorCountToMutate(
        _mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate
      )
      .setVectorMutationMethod(
        _mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod
      )
      .setMutationMode(_mainConfig.geneticConfig.kSampleConfig.mutationMode)
      .setMutationValue(_mainConfig.geneticConfig.kSampleConfig.mutationValue)
      .setLabelBalanceMode(
        _mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
      )
      .setCardinalityThreshold(
        _mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
      )
      .setNumericRatio(_mainConfig.geneticConfig.kSampleConfig.numericRatio)
      .setNumericTarget(_mainConfig.geneticConfig.kSampleConfig.numericTarget)
      .setTrainSplitChronologicalColumn(
        _mainConfig.geneticConfig.trainSplitChronologicalColumn
      )
      .setTrainSplitChronologicalRandomPercentage(
        _mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage
      )
      .setParallelism(_mainConfig.geneticConfig.parallelism)
      .setKFold(_mainConfig.geneticConfig.kFold)
      .setSeed(_mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(
        _mainConfig.geneticConfig.firstGenerationGenePool
      )
      .setNumberOfMutationGenerations(
        _mainConfig.geneticConfig.numberOfGenerations
      )
      .setNumberOfMutationsPerGeneration(
        _mainConfig.geneticConfig.numberOfMutationsPerGeneration
      )
      .setNumberOfParentsToRetain(
        _mainConfig.geneticConfig.numberOfParentsToRetain
      )
      .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(
        _mainConfig.geneticConfig.generationalMutationStrategy
      )
      .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
      .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
      .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
      .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
      .setContinuousEvolutionImprovementThreshold(
        _mainConfig.geneticConfig.continuousEvolutionImprovementThreshold
      )
      .setGeneticMBORegressorType(
        _mainConfig.geneticConfig.geneticMBORegressorType
      )
      .setGeneticMBOCandidateFactor(
        _mainConfig.geneticConfig.geneticMBOCandidateFactor
      )
      .setContinuousEvolutionMaxIterations(
        _mainConfig.geneticConfig.continuousEvolutionMaxIterations
      )
      .setContinuousEvolutionStoppingScore(
        _mainConfig.geneticConfig.continuousEvolutionStoppingScore
      )
      .setContinuousEvolutionParallelism(
        _mainConfig.geneticConfig.continuousEvolutionParallelism
      )
      .setContinuousEvolutionMutationAggressiveness(
        _mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness
      )
      .setContinuousEvolutionGeneticMixing(
        _mainConfig.geneticConfig.continuousEvolutionGeneticMixing
      )
      .setContinuousEvolutionRollingImporvementCount(
        _mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount
      )
      .setDataReductionFactor(_mainConfig.dataReductionFactor)
      .setFirstGenMode(_mainConfig.geneticConfig.initialGenerationMode)
      .setFirstGenPermutations(
        _mainConfig.geneticConfig.initialGenerationConfig.permutationCount
      )
      .setFirstGenIndexMixingMode(
        _mainConfig.geneticConfig.initialGenerationConfig.indexMixingMode
      )
      .setFirstGenArraySeed(
        _mainConfig.geneticConfig.initialGenerationConfig.arraySeed
      )
      .setHyperSpaceModelCount(_mainConfig.geneticConfig.hyperSpaceModelCount)

    if (_modelSeedSetStatus)
      initialize.setModelSeed(_mainConfig.geneticConfig.modelSeed)

    val (modelResultsRaw, modelStatsRaw) = initialize.evolveWithScoringDF()

    val resultBuffer = modelResultsRaw.toBuffer
    val statsBuffer = new ArrayBuffer[DataFrame]()
    statsBuffer += modelStatsRaw

    if (_mainConfig.geneticConfig.hyperSpaceInference) {

      println("\n\t\tStarting Post Tuning Inference Run.\n")

      val genericResults = new ArrayBuffer[GenericModelReturn]

      modelResultsRaw.foreach { x =>
        genericResults += GenericModelReturn(
          hyperParams = extractPayload(x.modelHyperParams),
          model = x.model,
          score = x.score,
          metrics = x.evalMetrics,
          generation = x.generation
        )
      }

      val hyperSpaceRunCandidates = new PostModelingOptimization()
        .setModelFamily("Trees")
        .setModelType(payload.modelType)
        .setHyperParameterSpaceCount(
          _mainConfig.geneticConfig.hyperSpaceInferenceCount
        )
        .setNumericBoundaries(_mainConfig.numericBoundaries)
        .setStringBoundaries(_mainConfig.stringBoundaries)
        .setSeed(_mainConfig.geneticConfig.seed)
        .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
        .treesPrediction(
          genericResults.result.toArray,
          _mainConfig.geneticConfig.hyperSpaceModelType,
          _mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =
        initialize.postRunModeledHyperParams(hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x
      }
      statsBuffer += hyperDataFrame

    }

    (
      resultBuffer.toArray,
      statsBuffer.reduce(_ union _),
      payload.modelType,
      cachedData
    )
  }

  private def logResultsToMlFlow(runData: Array[GenericModelReturn],
                                 modelFamily: String,
                                 modelType: String): MLFlowReportStructure = {

    val mlFlowLogger = MLFlowTracker(_mainConfig.mlFlowConfig)

    if (_mainConfig.mlFlowLogArtifactsFlag) mlFlowLogger.logArtifactsOn()
    else mlFlowLogger.logArtifactsOff()

    mlFlowLogger.logMlFlowDataAndModels(
      runData,
      modelFamily,
      modelType,
      _mainConfig.inferenceConfigSaveLocation,
      _mainConfig.scoringOptimizationStrategy
    )
  }

  private def logPipelineResultsToMlFlow(
    runData: Array[GenericModelReturn],
    modelFamily: String,
    modelType: String
  ): MLFlowReportStructure = {

    val mlFlowLogger = MLFlowTracker(_mainConfig.mlFlowConfig)
    mlFlowLogger.logMlFlowForPipeline(
      AutoMlPipelineMlFlowUtils
        .getMainConfigByPipelineId(_mainConfig.pipelineId)
        .mlFlowRunId,
      runData,
      modelFamily,
      modelType,
      _mainConfig.scoringOptimizationStrategy
    )
  }

  protected[automl] def executeTuning(
    payload: DataGeneration,
    isPipeline: Boolean = false
  ): TunerOutput = {

    val genericResults = new ArrayBuffer[GenericModelReturn]

    val (resultArray, modelStats, modelSelection, dataframe) =
      _mainConfig.modelFamily match {
        case "RandomForest" =>
          val (results, stats, selection, data) = runRandomForest(payload)
          results.foreach { x =>
            genericResults += GenericModelReturn(
              hyperParams = extractPayload(x.modelHyperParams),
              model = x.model,
              score = x.score,
              metrics = x.evalMetrics,
              generation = x.generation
            )
          }
          (genericResults, stats, selection, data)
        case "XGBoost" =>
          val (results, stats, selection, data) = runXGBoost(payload)
          results.foreach { x =>
            genericResults += GenericModelReturn(
              hyperParams = extractPayload(x.modelHyperParams),
              model = x.model,
              score = x.score,
              metrics = x.evalMetrics,
              generation = x.generation
            )
          }
          (genericResults, stats, selection, data)
        case "gbmBinary" | "gbmMulti" | "gbmMultiOVA" | "gbmHuber" | "gbmFair" |
            "gbmLasso" | "gbmRidge" | "gbmPoisson" | "gbmQuantile" | "gbmMape" |
            "gbmTweedie" | "gbmGamma" =>
          val (results, stats, selection, data) =
            runLightGBM(_mainConfig.modelFamily, payload)
          results.foreach { x =>
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
          val (results, stats, selection, data) = runGBT(payload)
          results.foreach { x =>
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
          val (results, stats, selection, data) = runMLPC(payload)
          results.foreach { x =>
            genericResults += GenericModelReturn(
              hyperParams = extractMLPCPayload(x.modelHyperParams),
              model = x.model,
              score = x.score,
              metrics = x.evalMetrics,
              generation = x.generation
            )
          }
          (genericResults, stats, selection, data)
        case "LinearRegression" =>
          val (results, stats, selection, data) = runLinearRegression(payload)
          results.foreach { x =>
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
          val (results, stats, selection, data) = runLogisticRegression(payload)
          results.foreach { x =>
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
          val (results, stats, selection, data) = runSVM(payload)
          results.foreach { x =>
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
          val (results, stats, selection, data) = runTrees(payload)
          results.foreach { x =>
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

    val mlFlow = if (_mainConfig.mlFlowLoggingFlag && !isPipeline) {

      // set the Inference details in general for the run
      // TODO - Remove this - It's here and in the tracker but the values are different and should be set equal
      val inferenceModelConfig = InferenceModelConfig(
        modelFamily = _mainConfig.modelFamily,
        modelType = modelSelection,
        modelLoadMethod = "path",
        mlFlowConfig = _mainConfig.mlFlowConfig,
        mlFlowRunId = "none",
        modelPathLocation = "notDefined"
      )

      // Set the Inference Config
      InferenceConfig.setInferenceModelConfig(inferenceModelConfig)
      InferenceConfig.setInferenceConfigStorageLocation(
        _mainConfig.inferenceConfigSaveLocation
      )

      // Write the Inference Payload out to the specified location
      val outputInferencePayload = InferenceConfig.getInferenceConfig

      val inferenceConfigReadable = convertInferenceConfigToJson(
        outputInferencePayload
      )
      val inferenceLog =
        s"Inference Configuration: \n${inferenceConfigReadable.prettyJson}"
      println(inferenceLog)

      logger.log(Level.INFO, inferenceLog)

      val mlFlowResult = try {
        logResultsToMlFlow(
          genericResultData,
          _mainConfig.modelFamily,
          modelSelection
        )
      } catch {
        case e: Exception =>
          println(
            s"Failed to log to mlflow.  Check configuration. \n ${e.printStackTrace()} " +
              s"\n ${e.getStackTraceString}"
          )
          logger.log(Level.FATAL, e.getStackTraceString)
          generateDummyMLFlowReturn("error").get
      }

      implicit val formats: Formats = Serialization.formats(hints = NoTypeHints)
      val pretty = writePretty(mlFlowResult)

      logger.log(Level.INFO, pretty)
      mlFlowResult
    } else if (isPipeline) {
      logPipelineResultsToMlFlow(
        genericResultData,
        _mainConfig.modelFamily,
        modelSelection
      )
    } else {
      generateDummyMLFlowReturn("undefined").get
    }

    val generationalData = extractGenerationalScores(
      genericResultData,
      _mainConfig.scoringOptimizationStrategy,
      _mainConfig.modelFamily,
      modelSelection
    )

    new TunerOutput(
      rawData = dataframe,
      modelSelection = modelSelection,
      mlFlowOutput = mlFlow
    ) {
      override def modelReport: Array[GenericModelReturn] = genericResultData
      override def generationReport: Array[GenerationalReport] =
        generationalData
      override def modelReportDataFrame: DataFrame = modelStats
      override def generationReportDataFrame: DataFrame =
        generationDataFrameReport(
          generationalData,
          _mainConfig.scoringOptimizationStrategy
        )
    }

  }

  private def generateDummyMLFlowReturn(
    msg: String
  ): Option[MLFlowReportStructure] = {
    try {
      val genTracker = MLFlowTracker(_mainConfig.mlFlowConfig)
      val dummyLog = MLFlowReturn(
        genTracker.createHostedMlFlowClient(),
        msg,
        Array((msg, 0.0))
      )
      Some(MLFlowReportStructure(dummyLog, dummyLog))
    } catch {
      case ex: Exception => Some(MLFlowReportStructure(null, null))
    }
  }

  protected[automl] def predictFromBestModel(
    resultPayload: Array[GenericModelReturn],
    rawData: DataFrame,
    modelSelection: String
  ): DataFrame = {

    val bestModel = resultPayload(0)

    _mainConfig.modelFamily match {
      case "RandomForest" =>
        modelSelection match {
          case "regressor" =>
            val model =
              bestModel.model.asInstanceOf[RandomForestRegressionModel]
            model.transform(rawData)
          case "classifier" =>
            val model =
              bestModel.model.asInstanceOf[RandomForestClassificationModel]
            model.transform(rawData)
        }
      case "XGBoost" =>
        modelSelection match {
          case "regressor" =>
            val model = bestModel.model.asInstanceOf[XGBoostRegressionModel]
            model.transform(rawData)
          case "classifier" =>
            val model = bestModel.model.asInstanceOf[XGBoostClassificationModel]
            model.transform(rawData)
        }
      case "gbmBinary" | "gbmMulti" | "gbmMultiOVA" =>
        val model = bestModel.model.asInstanceOf[LightGBMClassificationModel]
        model.transform(rawData)
      case "gbmHuber" | "gbmFair" | "gbmLasso" | "gbmRidge" | "gbmPoisson" |
          "gbmQuantile" | "gbmMape" | "gbmTweedie" | "gbmGamma" =>
        val model = bestModel.model.asInstanceOf[LightGBMRegressionModel]
        model.transform(rawData)
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
        val model =
          bestModel.model.asInstanceOf[MultilayerPerceptronClassificationModel]
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
          case "classifier" =>
            val model =
              bestModel.model.asInstanceOf[DecisionTreeClassificationModel]
            model.transform(rawData)
          case "regressor" =>
            val model =
              bestModel.model.asInstanceOf[DecisionTreeRegressionModel]
            model.transform(rawData)
        }
    }

  }

  @deprecated(
    "This method will be removed and replaced with the standalone version in " +
      "com.databricks.labs.automl.exploration.FeatureImportances in a future release."
  )
  def exploreFeatureImportances(): FeatureImportanceReturn = {

    println(
      "[DEPRECATION WARNING] - .exploreFeatureImportances() has been replaced by " +
        "com.databricks.labs.automl.exploration.FeatureImportances .  This method will be removed in the next release."
    )

    val payload = prepData()

    val cachedData = if (_featureImportancesConfig.dataPrepCachingFlag) {
      payload.data.persist(StorageLevel.MEMORY_AND_DISK)
      payload.data.count()
      payload.data
    } else {
      payload.data
    }

    if (_featureImportancesConfig.dataPrepCachingFlag) payload.data.count()

    val featureResults = new RandomForestFeatureImportance(
      cachedData,
      _featureImportancesConfig,
      payload.modelType
    ).setCutoffType(_featureImportancesConfig.featureImportanceCutoffType)
      .setCutoffValue(_featureImportancesConfig.featureImportanceCutoffValue)
      .runFeatureImportances(payload.fields)

    if (_featureImportancesConfig.dataPrepCachingFlag) cachedData.unpersist()

    FeatureImportanceReturn(
      featureResults._1,
      featureResults._2,
      featureResults._3,
      payload.modelType
    )
  }

  @deprecated(
    "This method will be removed and replaced with the standalone version in " +
      "com.databricks.labs.automl.exploration.FeatureImportances in a future release."
  )
  def runWithFeatureCulling(): FeatureImportanceOutput = {

    println(
      "[DEPRECATION WARNING] - .runWithFeatureCulling() has been replaced by " +
        "com.databricks.labs.automl.exploration.FeatureImportances .  This method will be removed in the next release."
    )

    // Get the Feature Importances

    val featureImportanceResults = exploreFeatureImportances()

    val selectableFields = featureImportanceResults.fields :+ _featureImportancesConfig.labelCol

    println(
      s"Feature Selected: ${featureImportanceResults.fields.mkString(", ")}"
    )

    val dataSubset = df.select(selectableFields.map(col): _*)

    if (_featureImportancesConfig.dataPrepCachingFlag) {
      dataSubset.persist(StorageLevel.MEMORY_AND_DISK)
      dataSubset.count
    }

    val runResults =
      new AutomationRunner(dataSubset).setMainConfig(_mainConfig).run()

    if (_mainConfig.dataPrepCachingFlag) dataSubset.unpersist()

    new FeatureImportanceOutput(
      featureImportanceResults.data,
      mlFlowOutput = runResults.mlFlowOutput
    ) {
      override def modelReport: Array[GenericModelReturn] =
        runResults.modelReport
      override def generationReport: Array[GenerationalReport] =
        runResults.generationReport
      override def modelReportDataFrame: DataFrame =
        runResults.modelReportDataFrame
      override def generationReportDataFrame: DataFrame =
        runResults.generationReportDataFrame
    }

  }

  @deprecated(
    "This method will be removed and replaced with the standalone version in " +
      "com.databricks.labs.automl.exploration.FeatureImportances in a future release."
  )
  def runFeatureCullingWithPrediction(): FeatureImportancePredictionOutput = {

    println(
      "[DEPRECATION WARNING] - .runFeatureCullingWithPrediction() has been replaced by " +
        "com.databricks.labs.automl.exploration.FeatureImportances .  This method will be removed in the next release."
    )

    val featureImportanceResults = exploreFeatureImportances()

    val selectableFields = featureImportanceResults.fields :+ _mainConfig.labelCol

    println(
      s"Features Selected: ${featureImportanceResults.fields.mkString(", ")}"
    )

    val dataSubset = df.select(selectableFields.map(col): _*)

    if (_mainConfig.dataPrepCachingFlag) {
      dataSubset.persist(StorageLevel.MEMORY_AND_DISK)
      dataSubset.count
    }

    val runner = new AutomationRunner(dataSubset).setMainConfig(_mainConfig)

    val payload = runner.prepData()

    val runResults = runner.executeTuning(payload)

    if (_mainConfig.dataPrepCachingFlag) dataSubset.unpersist()

    val predictedData = predictFromBestModel(
      runResults.modelReport,
      runResults.rawData,
      runResults.modelSelection
    )

    if (_mainConfig.dataPrepCachingFlag) runResults.rawData.unpersist()

    new FeatureImportancePredictionOutput(
      featureImportances = featureImportanceResults.data,
      predictionData = predictedData,
      mlFlowOutput = runResults.mlFlowOutput
    ) {
      override def modelReport: Array[GenericModelReturn] =
        runResults.modelReport
      override def generationReport: Array[GenerationalReport] =
        runResults.generationReport
      override def modelReportDataFrame: DataFrame =
        runResults.modelReportDataFrame
      override def generationReportDataFrame: DataFrame =
        runResults.generationReportDataFrame
    }

  }

  def generateDecisionSplits(): TreeSplitReport = {

    val payload = prepData()

    new DecisionTreeSplits(payload.data, _treeSplitsConfig, payload.modelType)
      .runTreeSplitAnalysis(payload.fields)

  }

  def run(): AutomationOutput = {

    val tunerResult = executeTuning(prepData())

    if (_mainConfig.dataPrepCachingFlag) tunerResult.rawData.unpersist()

    new AutomationOutput(mlFlowOutput = tunerResult.mlFlowOutput) {
      override def modelReport: Array[GenericModelReturn] =
        tunerResult.modelReport
      override def generationReport: Array[GenerationalReport] =
        tunerResult.generationReport
      override def modelReportDataFrame: DataFrame =
        tunerResult.modelReportDataFrame
      override def generationReportDataFrame: DataFrame =
        tunerResult.generationReportDataFrame
    }

  }

  def runWithPrediction(): PredictionOutput = {

    val tunerResult = executeTuning(prepData())

    val predictedData = predictFromBestModel(
      tunerResult.modelReport,
      tunerResult.rawData,
      tunerResult.modelSelection
    )

    if (_mainConfig.dataPrepCachingFlag) tunerResult.rawData.unpersist()

    new PredictionOutput(
      dataWithPredictions = predictedData,
      mlFlowOutput = tunerResult.mlFlowOutput
    ) {
      override def modelReport: Array[GenericModelReturn] =
        tunerResult.modelReport
      override def generationReport: Array[GenerationalReport] =
        tunerResult.generationReport
      override def modelReportDataFrame: DataFrame =
        tunerResult.modelReportDataFrame
      override def generationReportDataFrame: DataFrame =
        tunerResult.generationReportDataFrame
    }

  }

  def runWithConfusionReport(): ConfusionOutput = {
    val predictionPayload = runWithPrediction()
    val confusionData = predictionPayload.dataWithPredictions
      .select("prediction", _mainConfig.labelCol)
      .groupBy("prediction", _mainConfig.labelCol)
      .agg(count("*").alias("count"))

    new ConfusionOutput(
      predictionData = predictionPayload.dataWithPredictions,
      confusionData = confusionData,
      mlFlowOutput = predictionPayload.mlFlowOutput
    ) {
      override def modelReport: Array[GenericModelReturn] =
        predictionPayload.modelReport
      override def generationReport: Array[GenerationalReport] =
        predictionPayload.generationReport
      override def modelReportDataFrame: DataFrame =
        predictionPayload.modelReportDataFrame
      override def generationReportDataFrame: DataFrame =
        predictionPayload.generationReportDataFrame
    }

  }

}
