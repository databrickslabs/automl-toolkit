package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.AbstractTreeBinsSearchSpaceReset
import com.databricks.labs.automl.model.XGBoostTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._

import scala.collection.mutable.ArrayBuffer

private[tuner] class XGBoostTunerDelegator(mainConfig: MainConfig,
                            payload: DataGeneration,
                            testTrainSplitData: Array[TrainSplitReferences])
  extends AbstractTreeBinsSearchSpaceReset
     [XGBoostTuner,
      XGBoostModelsWithResults,
      XGBoostConfig,
      Any](mainConfig, payload, testTrainSplitData) {

  override protected def initializeTuner: XGBoostTuner = {
    val xgBoostTuner = new XGBoostTuner(payload.data, testTrainSplitData, payload.modelType, true)
      .setXGBoostNumericBoundaries(numericBoundaries.get)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(xgBoostTuner)
    xgBoostTuner
  }

  override protected def modelOptimization(tuner: XGBoostTuner,
                                             genericResults: ArrayBuffer[GenericModelReturn]):
  Array[XGBoostConfig] = {
    postModelingOptimization("XGBoost")
      .setNumericBoundaries(tuner.getXGBoostNumericBoundaries)
      .setStringBoundaries(mainConfig.stringBoundaries)
      .xgBoostPrediction(
        genericResults.result.toArray,
        mainConfig.geneticConfig.hyperSpaceModelType,
        mainConfig.geneticConfig.hyperSpaceModelCount
      )
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }
}

object XGBoostTunerDelegator {
  def apply(mainConfig: MainConfig, payload: DataGeneration, testTrainSplitData: Array[TrainSplitReferences]):
    XGBoostTunerDelegator = new XGBoostTunerDelegator(mainConfig, payload, testTrainSplitData)
}