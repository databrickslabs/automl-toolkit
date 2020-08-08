package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.GeneticTuner
import com.databricks.labs.automl.model.LightGBMTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, GenericModelReturn, LightGBMConfig, LightGBMModelsWithResults, MainConfig, TunerOutput}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class GbmTunerDelegator(mainConfig: MainConfig,
                        payload: DataGeneration,
                        testTrainSplitData: Array[TrainSplitReferences])
  extends GeneticTuner[LightGBMTuner, LightGBMModelsWithResults, LightGBMConfig](mainConfig, payload, testTrainSplitData) {


  override protected def initializeTuner: LightGBMTuner = {
    val lightGBMTuner = new LightGBMTuner(
      payload.data,
      testTrainSplitData,
      payload.modelType,
      mainConfig.modelFamily,
      true
    ).setLGBMNumericBoundaries(mainConfig.numericBoundaries)
      .setLGBMStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(lightGBMTuner)
    lightGBMTuner
  }

  override protected def delegateTuning(tuner: LightGBMTuner): TunerOutput = {
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

  override protected def hyperSpaceInference(tuner: LightGBMTuner,
                                             genericResults: ArrayBuffer[GenericModelReturn]):
  (ArrayBuffer[LightGBMModelsWithResults], ArrayBuffer[DataFrame]) = {
    val resultBuffer = new ArrayBuffer[LightGBMModelsWithResults]()
    val statsBuffer = new ArrayBuffer[DataFrame]()
    if (mainConfig.geneticConfig.hyperSpaceInference) {
      println("\n\t\tStarting Post Tuning Inference Run.\n")
      val hyperSpaceRunCandidates = postModelingOptimization(mainConfig.modelFamily)
        .setNumericBoundaries(tuner.getLightGBMNumericBoundaries)
        .setStringBoundaries(tuner.getLightGBMStringBoundaries)
        .lightGBMPrediction(
          genericResults.result.toArray,
          mainConfig.geneticConfig.hyperSpaceModelType,
          mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =  postRunModeledHyperParams(tuner,  hyperSpaceRunCandidates)

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
