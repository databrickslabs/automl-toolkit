package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.GeneticTuner
import com.databricks.labs.automl.model.MLPCTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class MlpcTunerDelegator(mainConfig: MainConfig,
                         payload: DataGeneration,
                         testTrainSplitData: Array[TrainSplitReferences])
  extends GeneticTuner[MLPCTuner, MLPCModelsWithResults, MLPCConfig](mainConfig, payload, testTrainSplitData) {


  override protected def initializeTuner: MLPCTuner = {
    val mlpcTuner = new MLPCTuner(payload.data, testTrainSplitData, true)
      .setMlpcNumericBoundaries(mainConfig.numericBoundaries)
      .setMlpcStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(mlpcTuner)
    mlpcTuner
  }

  override protected def delegateTuning(tuner: MLPCTuner): TunerOutput = {

    val (modelResultsRaw, modelStatsRaw) = payload.modelType match {
      case "classifier" => {
        evolve(tuner)
      }

      case _ =>
        throw new UnsupportedOperationException(
          s"Detected Model Type ${payload.modelType} is not supported by MultiLayer Perceptron Classifier"
        )
    }

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


  override protected def hyperSpaceInference(tuner: MLPCTuner,
                                                                    genericResults: ArrayBuffer[GenericModelReturn]):
  (ArrayBuffer[MLPCModelsWithResults], ArrayBuffer[DataFrame]) = {
    val resultBuffer = new ArrayBuffer[MLPCModelsWithResults]()
    val statsBuffer = new ArrayBuffer[DataFrame]()
    if (mainConfig.geneticConfig.hyperSpaceInference) {
      println("\n\t\tStarting Post Tuning Inference Run.\n")
      val hyperSpaceRunCandidates = postModelingOptimization("MLPC")
        .setNumericBoundaries(tuner.getMlpcNumericBoundaries)
        .setStringBoundaries(tuner.getMlpcStringBoundaries)
        .mlpcPrediction(
          genericResults.result.toArray,
          mainConfig.geneticConfig.hyperSpaceModelType,
          mainConfig.geneticConfig.hyperSpaceModelCount,
          tuner.getFeatureInputSize,
          tuner.getClassDistinctCount
        )

      val (hyperResults, hyperDataFrame) =  postRunModeledHyperParams(tuner, hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x.asInstanceOf[MLPCModelsWithResults]
      }
      statsBuffer += hyperDataFrame
    }

    (resultBuffer, statsBuffer)
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }
}
