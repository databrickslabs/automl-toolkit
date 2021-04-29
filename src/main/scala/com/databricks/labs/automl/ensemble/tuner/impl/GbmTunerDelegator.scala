package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.AbstractGeneticTunerDelegator
import com.databricks.labs.automl.model.LightGBMTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._

import scala.collection.mutable.ArrayBuffer

private[tuner] class GbmTunerDelegator(mainConfig: MainConfig,
                        payload: DataGeneration,
                        testTrainSplitData: Array[TrainSplitReferences])
  extends AbstractGeneticTunerDelegator
    [LightGBMTuner, LightGBMModelsWithResults, LightGBMConfig, Any](mainConfig, payload, testTrainSplitData) {

  override protected def initializeTuner: LightGBMTuner = {
    val lightGBMTuner = new LightGBMTuner(
      payload.data,
      testTrainSplitData,
      payload.modelType,
      mainConfig.modelFamily,
      true
    ).setLGBMNumericBoundaries(numericBoundaries.get)
      .setLGBMStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(lightGBMTuner)
    lightGBMTuner
  }

  override protected def modelOptimization(tuner: LightGBMTuner,
                                           genericResults: ArrayBuffer[GenericModelReturn]): Array[LightGBMConfig] = {
    postModelingOptimization(mainConfig.modelFamily)
      .setNumericBoundaries(tuner.getLightGBMNumericBoundaries)
      .setStringBoundaries(tuner.getLightGBMStringBoundaries)
      .lightGBMPrediction(
        genericResults.result.toArray,
        mainConfig.geneticConfig.hyperSpaceModelType,
        mainConfig.geneticConfig.hyperSpaceModelCount
      )
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }
}

object GbmTunerDelegator {
  def apply(mainConfig: MainConfig, payload: DataGeneration, testTrainSplitData: Array[TrainSplitReferences]):
    GbmTunerDelegator = new GbmTunerDelegator(mainConfig, payload, testTrainSplitData)
}
