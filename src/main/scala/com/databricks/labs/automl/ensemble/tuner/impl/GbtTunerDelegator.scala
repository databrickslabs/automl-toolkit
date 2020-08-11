package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.AbstractTreeBinsSearchSpaceReset
import com.databricks.labs.automl.model.GBTreesTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{GBTModelsWithResults, _}

import scala.collection.mutable.ArrayBuffer

private[tuner] class GbtTunerDelegator(mainConfig: MainConfig,
                        payload: DataGeneration,
                        testTrainSplitData: Array[TrainSplitReferences])
  extends AbstractTreeBinsSearchSpaceReset
     [GBTreesTuner,
      GBTModelsWithResults,
      GBTConfig,
      Any](mainConfig, payload, testTrainSplitData) {

  override protected def initializeTuner: GBTreesTuner = {
    val gbTreesTuner = new GBTreesTuner(payload.data, testTrainSplitData, payload.modelType, true)
      .setGBTNumericBoundaries(numericBoundaries.get)
      .setGBTStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(gbTreesTuner)
    gbTreesTuner
  }

  override protected def modelOptimization(tuner: GBTreesTuner,
                                           genericResults: ArrayBuffer[GenericModelReturn]): Array[GBTConfig] = {
    postModelingOptimization(mainConfig.modelFamily)
      .setNumericBoundaries(tuner.getGBTNumericBoundaries)
      .setStringBoundaries(tuner.getGBTStringBoundaries)
      .gbtPrediction(
        genericResults.result.toArray,
        mainConfig.geneticConfig.hyperSpaceModelType,
        mainConfig.geneticConfig.hyperSpaceModelCount
      )
  }

  override def validate(mainConfig: MainConfig): Unit = {
    
  }
}

object GbtTunerDelegator {
  def apply(mainConfig: MainConfig, payload: DataGeneration, testTrainSplitData: Array[TrainSplitReferences]):
    GbtTunerDelegator = new GbtTunerDelegator(mainConfig, payload, testTrainSplitData)
}
