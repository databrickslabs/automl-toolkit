package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.ensemble.tuner.validate.GeneticTunerValidator
import com.databricks.labs.automl.model.tools.PostModelingOptimization
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.model.{AbstractTuner, Evolution}
import com.databricks.labs.automl.params.{TunerConfigBase, TunerOutputWithResults, _}
import com.databricks.labs.automl.utils.AutomationTools
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

//import scala.collection.mutable.ArrayBuffer

abstract class AbstractGeneticTunerDelegator[A <: AbstractTuner[C, B, D],
                            B <: TunerOutputWithResults[C, D],
                            C <: TunerConfigBase,
                            D](mainConfig: MainConfig,
                               payload: DataGeneration,
                               testTrainSplitData: Array[TrainSplitReferences])
  extends GeneticTunerValidator
    with AutomationTools
    with TunerDelegator {

  override def tune: TunerOutput = {
    validate(mainConfig)
    delegateTuning
  }

  protected def initializeTuner: A

  override def numericBoundaries: Option[Map[String, (Double, Double)]] = {
    Some(mainConfig.numericBoundaries)
  }

  protected def delegateTuning: TunerOutput = {
    val tuner = initializeTuner
    val (modelResultsRaw, modelStatsRaw) = evolve(tuner)
    val resultBuffer = modelResultsRaw.toBuffer
    val statsBuffer = new ArrayBuffer[DataFrame]()
    statsBuffer += modelStatsRaw

    val genericResults = modelResultsRaw.map(item => GenericModelReturn(
        hyperParams = extractPayload(item.modelHyperParams),
        model = item.model,
        score = item.score,
        metrics = item.evalMetrics,
        generation = item.generation
      ))

    val genResultsAsBuffer = genericResults.to[ArrayBuffer]

    val (resultBuffer1, statsBuffer1) = hyperSpaceInference(tuner, genResultsAsBuffer)
    statsBuffer ++= statsBuffer1
    resultBuffer ++= resultBuffer1

    tunerOutput(
      statsBuffer.reduce(_ union _),
      payload.modelType,
      payload.data,
      genericResults.toArray
    )
  }

  protected def evolve(tuner: A): (Array[B], DataFrame) = {
    val evolveResult = tuner.evolveWithScoringDF()
    (evolveResult._1.asInstanceOf[Array[B]], evolveResult._2)
  }

  protected def hyperSpaceInference(tuner: A,
                                    genericResults: ArrayBuffer[GenericModelReturn]):
  (ArrayBuffer[B], ArrayBuffer[DataFrame]) = {
    val resultBuffer = new ArrayBuffer[B]()
    val statsBuffer = new ArrayBuffer[DataFrame]()
    if (mainConfig.geneticConfig.hyperSpaceInference) {
      println("\n\t\tStarting Post Tuning Inference Run.\n")

      val hyperSpaceRunCandidates = modelOptimization(tuner, genericResults)

      val (hyperResults, hyperDataFrame) =  postRunModeledHyperParams(tuner, hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x
      }
      statsBuffer += hyperDataFrame
    }

    (resultBuffer, statsBuffer)
  }

  protected def modelOptimization(tuner: A,
                                  genericResults: ArrayBuffer[GenericModelReturn]): Array[C]

  protected def postRunModeledHyperParams(tuner: A,
                                          hyperSpaceRunCandidates: Array[C]):
  (Array[B], DataFrame) = {
    val retTuple = tuner.postRunModeledHyperParams(hyperSpaceRunCandidates)
    (retTuple._1.asInstanceOf[Array[B]], retTuple._2)
  }

  def setTunerEvolutionConfig(evolutionConfig: Evolution): Unit = {
    evolutionConfig
      .setLabelCol(mainConfig.labelCol)
      .setFeaturesCol(mainConfig.featuresCol)
      .setFieldsToIgnore(mainConfig.fieldsToIgnoreInVector)
      .setTrainPortion(mainConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(
        trainSplitValidation(
          mainConfig.geneticConfig.trainSplitMethod,
          payload.modelType
        )
      )
      .setSyntheticCol(mainConfig.geneticConfig.kSampleConfig.syntheticCol)
      .setKGroups(mainConfig.geneticConfig.kSampleConfig.kGroups)
      .setKMeansMaxIter(
        mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter
      )
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
      .setLSHHashTables(
        mainConfig.geneticConfig.kSampleConfig.lshHashTables
      )
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
      .setMutationValue(
        mainConfig.geneticConfig.kSampleConfig.mutationValue
      )
      .setLabelBalanceMode(
        mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
      )
      .setCardinalityThreshold(
        mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
      )
      .setNumericRatio(mainConfig.geneticConfig.kSampleConfig.numericRatio)
      .setNumericTarget(
        mainConfig.geneticConfig.kSampleConfig.numericTarget
      )
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
      .setNumberOfMutationsPerGeneration(
        mainConfig.geneticConfig.numberOfMutationsPerGeneration
      )
      .setGeneticMixing(mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(
        mainConfig.geneticConfig.generationalMutationStrategy
      )
      .setMutationMagnitudeMode(
        mainConfig.geneticConfig.mutationMagnitudeMode
      )
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
      .setHyperSpaceModelCount(
        mainConfig.geneticConfig.hyperSpaceModelCount
      )
      .setModelSeed(mainConfig.geneticConfig.modelSeed)
  }


  def postModelingOptimization(modelFamily: String): PostModelingOptimization = {
    new PostModelingOptimization()
      .setModelFamily(modelFamily)
      .setModelType(payload.modelType)
      .setHyperParameterSpaceCount(
        mainConfig.geneticConfig.hyperSpaceInferenceCount
      )
      .setSeed(mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(mainConfig.scoringOptimizationStrategy)
      .setSeed(mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(mainConfig.scoringOptimizationStrategy)
  }

  def tunerOutput(modelStats: DataFrame,
                  modelSelection: String,
                  dataframe: DataFrame,
                  genericResultData: Array[GenericModelReturn]): TunerOutput = {

    val generationalData = extractGenerationalScores(
      genericResultData,
      mainConfig.scoringOptimizationStrategy,
      mainConfig.modelFamily,
      modelSelection
    )

    val mlFlow = if (mainConfig.mlFlowLoggingFlag) {
      logPipelineResultsToMlFlow(
        genericResultData,
        mainConfig.modelFamily,
        modelSelection,
        mainConfig
      )
    } else {
      generateDummyMLFlowReturn("undefined", mainConfig).get
    }

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
          mainConfig.scoringOptimizationStrategy
        )
    }
  }

}
