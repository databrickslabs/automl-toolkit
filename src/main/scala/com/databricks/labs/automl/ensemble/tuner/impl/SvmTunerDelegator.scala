package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.GeneticTuner
import com.databricks.labs.automl.model.SVMTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class SvmTunerDelegator(mainConfig: MainConfig,
                        payload: DataGeneration,
                        testTrainSplitData: Array[TrainSplitReferences])
  extends GeneticTuner[SVMTuner, SVMModelsWithResults, SVMConfig](mainConfig, payload, testTrainSplitData) {


  override protected def initializeTuner: SVMTuner = {
    val svmTuner = new SVMTuner(
      payload.data,
      testTrainSplitData,
      true)
      .setSvmNumericBoundaries(mainConfig.numericBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(svmTuner)
    svmTuner
  }

  override protected def delegateTuning(tuner: SVMTuner): TunerOutput = {

    val (modelResultsRaw, modelStatsRaw) = payload.modelType match {
      case "classifier" => {
        evolve(tuner)
      }

      case _ =>
        throw new UnsupportedOperationException(
          s"Detected Model Type ${payload.modelType} is not supported by Support Vector Machines"
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

  override protected def hyperSpaceInference(tuner: SVMTuner,
                                                                   genericResults: ArrayBuffer[GenericModelReturn]):
  (ArrayBuffer[SVMModelsWithResults], ArrayBuffer[DataFrame]) = {
    val resultBuffer = new ArrayBuffer[SVMModelsWithResults]()
    val statsBuffer = new ArrayBuffer[DataFrame]()
    if (mainConfig.geneticConfig.hyperSpaceInference) {
      println("\n\t\tStarting Post Tuning Inference Run.\n")
      val hyperSpaceRunCandidates = postModelingOptimization("SVM")
        .setNumericBoundaries(tuner.getSvmNumericBoundaries)
        .setStringBoundaries(mainConfig.stringBoundaries)
        .svmPrediction(
          genericResults.result.toArray,
          mainConfig.geneticConfig.hyperSpaceModelType,
          mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =  postRunModeledHyperParams(tuner, hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x.asInstanceOf[SVMModelsWithResults]
      }
      statsBuffer += hyperDataFrame
    }

    (resultBuffer, statsBuffer)
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }
}
