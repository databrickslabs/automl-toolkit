package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.AbstractTreeBinsSearchSpaceReset
import com.databricks.labs.automl.model.DecisionTreeTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._

import scala.collection.mutable.ArrayBuffer

private[tuner] class TreesTunerDelegator(mainConfig: MainConfig,
                          payload: DataGeneration,
                          testTrainSplitData: Array[TrainSplitReferences])
  extends AbstractTreeBinsSearchSpaceReset
     [DecisionTreeTuner,
      TreesModelsWithResults,
      TreesConfig,
      Any](mainConfig, payload, testTrainSplitData) {

  override protected def initializeTuner: DecisionTreeTuner = {
    val decisionTreeTuner = new DecisionTreeTuner(payload.data, testTrainSplitData, payload.modelType, true)
      .setTreesNumericBoundaries(numericBoundaries.get)
      .setTreesStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(decisionTreeTuner)
    decisionTreeTuner
  }

  override protected def modelOptimization(tuner: DecisionTreeTuner,
                                           genericResults: ArrayBuffer[GenericModelReturn]): Array[TreesConfig] = {
    postModelingOptimization("RandomForest")
      .setNumericBoundaries(tuner.getTreesNumericBoundaries)
      .setStringBoundaries(tuner.getTreesStringBoundaries)
      .treesPrediction(
        genericResults.result.toArray,
        mainConfig.geneticConfig.hyperSpaceModelType,
        mainConfig.geneticConfig.hyperSpaceModelCount
      )
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }
}

object TreesTunerDelegator {
  def apply(mainConfig: MainConfig, payload: DataGeneration, testTrainSplitData: Array[TrainSplitReferences]):
    TreesTunerDelegator = new TreesTunerDelegator(mainConfig, payload, testTrainSplitData)
}
