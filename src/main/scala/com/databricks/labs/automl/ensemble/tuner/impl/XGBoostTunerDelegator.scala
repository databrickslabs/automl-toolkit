package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.GeneticTuner
import com.databricks.labs.automl.model.XGBoostTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class XGBoostTunerDelegator(mainConfig: MainConfig,
                            payload: DataGeneration,
                            testTrainSplitData: Array[TrainSplitReferences])
  extends GeneticTuner[XGBoostTuner, XGBoostModelsWithResults, XGBoostConfig](mainConfig, payload, testTrainSplitData) {

  override protected def initializeTuner: XGBoostTuner = {
    val xgBoostTuner = new XGBoostTuner(payload.data, testTrainSplitData, payload.modelType, true)
      .setXGBoostNumericBoundaries(mainConfig.numericBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(xgBoostTuner)
    xgBoostTuner
  }

  override protected def delegateTuning(tuner: XGBoostTuner): TunerOutput = {

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

  override protected def hyperSpaceInference(tuner: XGBoostTuner,
                                                                     genericResults: ArrayBuffer[GenericModelReturn]):
  (ArrayBuffer[XGBoostModelsWithResults], ArrayBuffer[DataFrame]) = {
    val resultBuffer = new ArrayBuffer[XGBoostModelsWithResults]()
    val statsBuffer = new ArrayBuffer[DataFrame]()
    if (mainConfig.geneticConfig.hyperSpaceInference) {
      println("\n\t\tStarting Post Tuning Inference Run.\n")
      val hyperSpaceRunCandidates = postModelingOptimization("XGBoost")
        .setNumericBoundaries(tuner.getXGBoostNumericBoundaries)
        .setStringBoundaries(mainConfig.stringBoundaries)
        .xgBoostPrediction(
          genericResults.result.toArray,
          mainConfig.geneticConfig.hyperSpaceModelType,
          mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =  postRunModeledHyperParams(tuner, hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x.asInstanceOf[XGBoostModelsWithResults]
      }
      statsBuffer += hyperDataFrame
    }

    (resultBuffer, statsBuffer)
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }
}
