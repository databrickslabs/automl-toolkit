package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.GeneticTuner
import com.databricks.labs.automl.ensemble.tuner.exception.TuningException
import com.databricks.labs.automl.model.RandomForestTuner
import com.databricks.labs.automl.model.tools.PostModelingOptimization
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, GenericModelReturn, MainConfig, TunerOutput}
import com.databricks.labs.automl.utils.AutomationTools
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class RandomForestTunerDelegator extends GeneticTuner {

  override protected def delegateTuning(mainConfig: MainConfig,
                                        payload: DataGeneration,
                                        testTrainSplitData: Array[TrainSplitReferences]): TunerOutput = {

    val initialize = new RandomForestTuner(
      payload.data,
      testTrainSplitData,
      payload.modelType,
      true
    ).setLabelCol(mainConfig.labelCol)
      .setFeaturesCol(mainConfig.featuresCol)
      .setFieldsToIgnore(mainConfig.fieldsToIgnoreInVector)
      .setRandomForestNumericBoundaries(mainConfig.numericBoundaries)
      .setRandomForestStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
      .setTrainPortion(mainConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(
        trainSplitValidation(
          mainConfig.geneticConfig.trainSplitMethod,
          payload.modelType
        )
      )
      .setSyntheticCol(mainConfig.geneticConfig.kSampleConfig.syntheticCol)
      .setKGroups(mainConfig.geneticConfig.kSampleConfig.kGroups)
      .setKMeansMaxIter(mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter)
      .setKMeansTolerance(
        mainConfig.geneticConfig.kSampleConfig.kMeansTolerance
      )
      .setKMeansDistanceMeasurement(
        mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement
      )
      .setKMeansSeed(mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
      .setKMeansPredictionCol(
        mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol
      )
      .setLSHHashTables(mainConfig.geneticConfig.kSampleConfig.lshHashTables)
      .setLSHSeed(mainConfig.geneticConfig.kSampleConfig.lshSeed)
      .setLSHOutputCol(mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
      .setQuorumCount(mainConfig.geneticConfig.kSampleConfig.quorumCount)
      .setMinimumVectorCountToMutate(
        mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate
      )
      .setVectorMutationMethod(
        mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod
      )
      .setMutationMode(mainConfig.geneticConfig.kSampleConfig.mutationMode)
      .setMutationValue(mainConfig.geneticConfig.kSampleConfig.mutationValue)
      .setLabelBalanceMode(
        mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
      )
      .setCardinalityThreshold(
        mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
      )
      .setNumericRatio(mainConfig.geneticConfig.kSampleConfig.numericRatio)
      .setNumericTarget(mainConfig.geneticConfig.kSampleConfig.numericTarget)
      .setTrainSplitChronologicalColumn(
        mainConfig.geneticConfig.trainSplitChronologicalColumn
      )
      .setTrainSplitChronologicalRandomPercentage(
        mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage
      )
      .setParallelism(mainConfig.geneticConfig.parallelism)
      .setKFold(mainConfig.geneticConfig.kFold)
      .setSeed(mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(
        mainConfig.geneticConfig.firstGenerationGenePool
      )
      .setNumberOfMutationGenerations(
        mainConfig.geneticConfig.numberOfGenerations
      )
      .setNumberOfMutationsPerGeneration(
        mainConfig.geneticConfig.numberOfMutationsPerGeneration
      )
      .setNumberOfParentsToRetain(
        mainConfig.geneticConfig.numberOfParentsToRetain
      )
      .setGeneticMixing(mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(
        mainConfig.geneticConfig.generationalMutationStrategy
      )
      .setMutationMagnitudeMode(mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(mainConfig.geneticConfig.fixedMutationValue)
      .setEarlyStoppingFlag(mainConfig.autoStoppingFlag)
      .setEarlyStoppingScore(mainConfig.autoStoppingScore)
      .setEvolutionStrategy(mainConfig.geneticConfig.evolutionStrategy)
      .setContinuousEvolutionImprovementThreshold(
        mainConfig.geneticConfig.continuousEvolutionImprovementThreshold
      )
      .setGeneticMBORegressorType(
        mainConfig.geneticConfig.geneticMBORegressorType
      )
      .setGeneticMBOCandidateFactor(
        mainConfig.geneticConfig.geneticMBOCandidateFactor
      )
      .setContinuousEvolutionMaxIterations(
        mainConfig.geneticConfig.continuousEvolutionMaxIterations
      )
      .setContinuousEvolutionStoppingScore(
        mainConfig.geneticConfig.continuousEvolutionStoppingScore
      )
      .setContinuousEvolutionParallelism(
        mainConfig.geneticConfig.continuousEvolutionParallelism
      )
      .setContinuousEvolutionMutationAggressiveness(
        mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness
      )
      .setContinuousEvolutionGeneticMixing(
        mainConfig.geneticConfig.continuousEvolutionGeneticMixing
      )
      .setContinuousEvolutionRollingImporvementCount(
        mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount
      )
      .setDataReductionFactor(mainConfig.dataReductionFactor)
      .setFirstGenMode(mainConfig.geneticConfig.initialGenerationMode)
      .setFirstGenPermutations(
        mainConfig.geneticConfig.initialGenerationConfig.permutationCount
      )
      .setFirstGenIndexMixingMode(
        mainConfig.geneticConfig.initialGenerationConfig.indexMixingMode
      )
      .setFirstGenArraySeed(
        mainConfig.geneticConfig.initialGenerationConfig.arraySeed
      )
      .setHyperSpaceModelCount(mainConfig.geneticConfig.hyperSpaceModelCount)

      initialize.setModelSeed(mainConfig.geneticConfig.modelSeed)

    val (modelResultsRaw, modelStatsRaw) = initialize.evolveWithScoringDF()

    val resultBuffer = modelResultsRaw.toBuffer
    val statsBuffer = new ArrayBuffer[DataFrame]()
    statsBuffer += modelStatsRaw

    if (mainConfig.geneticConfig.hyperSpaceInference) {

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
          mainConfig.geneticConfig.hyperSpaceInferenceCount
        )
        .setNumericBoundaries(initialize.getRandomForestNumericBoundaries)
        .setStringBoundaries(initialize.getRandomForestStringBoundaries)
        .setSeed(mainConfig.geneticConfig.seed)
        .setOptimizationStrategy(mainConfig.scoringOptimizationStrategy)
        .randomForestPrediction(
          genericResults.result.toArray,
          mainConfig.geneticConfig.hyperSpaceModelType,
          mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =
        initialize.postRunModeledHyperParams(hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x
      }
      statsBuffer += hyperDataFrame

    }

    tunerOutput(
      resultBuffer.toArray,
      statsBuffer.reduce(_ union _),
      payload.modelType,
      payload.data
    )
  }

  override def validate(mainConfig: MainConfig): Unit = {
    "d"
  }

}
