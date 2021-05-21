package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.AbstractTreeBinsSearchSpaceReset
import com.databricks.labs.automl.model.RandomForestTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._

import scala.collection.mutable.ArrayBuffer

private[tuner] class RandomForestTunerDelegator(mainConfig: MainConfig,
                                 payload: DataGeneration,
                                 testTrainSplitData: Array[TrainSplitReferences])
  extends AbstractTreeBinsSearchSpaceReset
     [RandomForestTuner,
      RandomForestModelsWithResults,
      RandomForestConfig,
      Any](mainConfig, payload, testTrainSplitData) {

  override protected def initializeTuner: RandomForestTuner = {
    val randomForestTuner = new RandomForestTuner(payload.data, testTrainSplitData, payload.modelType, true)
      .setRandomForestNumericBoundaries(numericBoundaries.get)
      .setRandomForestStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(randomForestTuner)
    randomForestTuner
  }

  override protected def modelOptimization(tuner: RandomForestTuner,
                                           genericResults: ArrayBuffer[GenericModelReturn]): Array[RandomForestConfig] = {
    postModelingOptimization("RandomForest")
      .setNumericBoundaries(tuner.getRandomForestNumericBoundaries)
      .setStringBoundaries(tuner.getRandomForestStringBoundaries)
      .randomForestPrediction(
        genericResults.result.toArray,
        mainConfig.geneticConfig.hyperSpaceModelType,
        mainConfig.geneticConfig.hyperSpaceModelCount
      )
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }

}

object RandomForestTunerDelegator {
  def apply(mainConfig: MainConfig, payload: DataGeneration, testTrainSplitData: Array[TrainSplitReferences]):
    RandomForestTunerDelegator = new RandomForestTunerDelegator(mainConfig, payload, testTrainSplitData)
}