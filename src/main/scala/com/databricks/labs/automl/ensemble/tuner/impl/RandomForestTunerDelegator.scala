package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.GeneticTuner
import com.databricks.labs.automl.model.RandomForestTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class RandomForestTunerDelegator(mainConfig: MainConfig,
                                 payload: DataGeneration,
                                 testTrainSplitData: Array[TrainSplitReferences])
  extends GeneticTuner[RandomForestTuner, RandomForestModelsWithResults, RandomForestConfig](mainConfig, payload, testTrainSplitData) {


  override protected def initializeTuner: RandomForestTuner = {
    val randomForestTuner = new RandomForestTuner(payload.data, testTrainSplitData, payload.modelType, true)
      .setRandomForestNumericBoundaries(mainConfig.numericBoundaries)
      .setRandomForestStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(randomForestTuner)
    randomForestTuner
  }

  override protected def delegateTuning(tuner: RandomForestTuner): TunerOutput = {

    val (modelResultsRaw, modelStatsRaw) = evolve(tuner)

    val resultBuffer = modelResultsRaw.toBuffer
    val statsBuffer = new ArrayBuffer[DataFrame]()
    statsBuffer += modelStatsRaw

    val genericResults = modelResultsRaw.map(item => {
      GenericModelReturn(
        hyperParams = extractPayload(item.modelHyperParams),
        model = item.model,
        score = item.score,
        metrics = item.evalMetrics,
        generation = item.generation
      )
    }).asInstanceOf[ArrayBuffer[GenericModelReturn]]

    val (resultBuffer1, statsBuffer1) = hyperSpaceInference(tuner, genericResults)
    statsBuffer ++= statsBuffer1
    resultBuffer ++= resultBuffer1

    tunerOutput(
      statsBuffer.reduce(_ union _),
      payload.modelType,
      payload.data,
      genericResults.toArray
    )
  }

  override protected def hyperSpaceInference(tuner: RandomForestTuner,
                                                                            genericResults: ArrayBuffer[GenericModelReturn]):
  (ArrayBuffer[RandomForestModelsWithResults], ArrayBuffer[DataFrame]) = {
    val resultBuffer = new ArrayBuffer[RandomForestModelsWithResults]()
    val statsBuffer = new ArrayBuffer[DataFrame]()
    if (mainConfig.geneticConfig.hyperSpaceInference) {
      println("\n\t\tStarting Post Tuning Inference Run.\n")
      val hyperSpaceRunCandidates = postModelingOptimization("RandomForest")
        .setNumericBoundaries(tuner.getRandomForestNumericBoundaries)
        .setStringBoundaries(tuner.getRandomForestStringBoundaries)
        .randomForestPrediction(
          genericResults.result.toArray,
          mainConfig.geneticConfig.hyperSpaceModelType,
          mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =  postRunModeledHyperParams(tuner, hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x.asInstanceOf[RandomForestModelsWithResults]
      }
      statsBuffer += hyperDataFrame
    }

    (resultBuffer, statsBuffer)
  }

  override def validate(mainConfig: MainConfig): Unit = {
    "d"
  }

}
