package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.GeneticTuner
import com.databricks.labs.automl.model.GBTreesTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{GBTModelsWithResults, _}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class GbtTunerDelegator(mainConfig: MainConfig,
                        payload: DataGeneration,
                        testTrainSplitData: Array[TrainSplitReferences])
  extends GeneticTuner[GBTreesTuner, GBTModelsWithResults, GBTConfig](mainConfig, payload, testTrainSplitData) {

  override protected def initializeTuner: GBTreesTuner = {
    val gbTreesTuner = new GBTreesTuner(payload.data, testTrainSplitData, payload.modelType, true)
      .setGBTNumericBoundaries(mainConfig.numericBoundaries)
      .setGBTStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(gbTreesTuner)
    gbTreesTuner
  }

  override protected def delegateTuning(tuner: GBTreesTuner): TunerOutput = {

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

  override protected def hyperSpaceInference(tuner: GBTreesTuner,
                                             genericResults: ArrayBuffer[GenericModelReturn]):
  (ArrayBuffer[GBTModelsWithResults], ArrayBuffer[DataFrame]) = {
    val resultBuffer = new ArrayBuffer[GBTModelsWithResults]()
    val statsBuffer = new ArrayBuffer[DataFrame]()
    if (mainConfig.geneticConfig.hyperSpaceInference) {
      println("\n\t\tStarting Post Tuning Inference Run.\n")
      val hyperSpaceRunCandidates = postModelingOptimization(mainConfig.modelFamily)
        .setNumericBoundaries(tuner.getGBTNumericBoundaries)
        .setStringBoundaries(tuner.getGBTStringBoundaries)
        .gbtPrediction(
          genericResults.result.toArray,
          mainConfig.geneticConfig.hyperSpaceModelType,
          mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =  postRunModeledHyperParams(tuner, hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x
      }
      statsBuffer += hyperDataFrame
    }

    (resultBuffer, statsBuffer)

  }

  override def validate(mainConfig: MainConfig): Unit = {
    
  }
}
