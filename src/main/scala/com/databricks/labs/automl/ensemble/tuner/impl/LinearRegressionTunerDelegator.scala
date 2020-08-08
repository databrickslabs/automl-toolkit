package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.GeneticTuner
import com.databricks.labs.automl.model.LinearRegressionTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class LinearRegressionTunerDelegator(mainConfig: MainConfig,
                                     payload: DataGeneration,
                                     testTrainSplitData: Array[TrainSplitReferences])
  extends GeneticTuner[LinearRegressionTuner, LinearRegressionModelsWithResults, LinearRegressionConfig](mainConfig, payload, testTrainSplitData) {


  override protected def initializeTuner: LinearRegressionTuner = {
    val linearRegressionTuner = new LinearRegressionTuner(
      payload.data,
      testTrainSplitData,
      true
    ).setLinearRegressionNumericBoundaries(mainConfig.numericBoundaries)
      .setLinearRegressionStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(linearRegressionTuner)
    linearRegressionTuner
  }

  override protected def delegateTuning(tuner: LinearRegressionTuner): TunerOutput = {

    val (modelResultsRaw, modelStatsRaw) = payload.modelType match {
      case "regressor" => {
          evolve(tuner)
      }

      case _ =>
        throw new UnsupportedOperationException(
          s"Detected Model Type ${payload.modelType} is not supported by Linear Regression"
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

  override protected def hyperSpaceInference(tuner: LinearRegressionTuner,
                                             genericResults: ArrayBuffer[GenericModelReturn]):
  (ArrayBuffer[LinearRegressionModelsWithResults], ArrayBuffer[DataFrame]) = {
    val resultBuffer = new ArrayBuffer[LinearRegressionModelsWithResults]()
    val statsBuffer = new ArrayBuffer[DataFrame]()
    if (mainConfig.geneticConfig.hyperSpaceInference) {
      println("\n\t\tStarting Post Tuning Inference Run.\n")
      val hyperSpaceRunCandidates = postModelingOptimization("LinearRegression")
        .setNumericBoundaries(tuner.getLinearRegressionNumericBoundaries)
        .setStringBoundaries(tuner.getLinearRegressionStringBoundaries)
        .linearRegressionPrediction(
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
