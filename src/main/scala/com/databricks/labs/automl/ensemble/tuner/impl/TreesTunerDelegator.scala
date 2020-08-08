package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.GeneticTuner
import com.databricks.labs.automl.model.{DecisionTreeTuner, SVMTuner}
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, GenericModelReturn, LightGBMModelsWithResults, MainConfig, TreesConfig, TreesModelsWithResults, TunerOutput}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class TreesTunerDelegator(mainConfig: MainConfig,
                          payload: DataGeneration,
                          testTrainSplitData: Array[TrainSplitReferences])
  extends GeneticTuner[DecisionTreeTuner, TreesModelsWithResults, TreesConfig](mainConfig, payload, testTrainSplitData) {


  override protected def initializeTuner: DecisionTreeTuner = {
    val decisionTreeTuner = new DecisionTreeTuner(payload.data, testTrainSplitData, payload.modelType, true)
      .setTreesNumericBoundaries(mainConfig.numericBoundaries)
      .setTreesStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(decisionTreeTuner)
    decisionTreeTuner
  }

  override protected def delegateTuning(tuner: DecisionTreeTuner): TunerOutput = {

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

  override protected def hyperSpaceInference(tuner: DecisionTreeTuner,
                                                                   genericResults: ArrayBuffer[GenericModelReturn]):
  (ArrayBuffer[TreesModelsWithResults], ArrayBuffer[DataFrame]) = {
    val resultBuffer = new ArrayBuffer[TreesModelsWithResults]()
    val statsBuffer = new ArrayBuffer[DataFrame]()
    if (mainConfig.geneticConfig.hyperSpaceInference) {
      println("\n\t\tStarting Post Tuning Inference Run.\n")
      val hyperSpaceRunCandidates = postModelingOptimization("RandomForest")
        .setNumericBoundaries(tuner.getTreesNumericBoundaries)
        .setStringBoundaries(tuner.getTreesStringBoundaries)
        .treesPrediction(
          genericResults.result.toArray,
          mainConfig.geneticConfig.hyperSpaceModelType,
          mainConfig.geneticConfig.hyperSpaceModelCount
        )

      val (hyperResults, hyperDataFrame) =  postRunModeledHyperParams(tuner, hyperSpaceRunCandidates)

      hyperResults.foreach { x =>
        resultBuffer += x.asInstanceOf[TreesModelsWithResults]
      }
      statsBuffer += hyperDataFrame
    }

    (resultBuffer, statsBuffer)
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }
}
