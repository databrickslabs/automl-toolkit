package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.AbstractGeneticTunerDelegator
import com.databricks.labs.automl.model.{DecisionTreeTuner, SVMTuner}
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, GenericModelReturn, LightGBMModelsWithResults, MainConfig, TreesConfig, TreesModelsWithResults, TunerOutput}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

private[tuner] class TreesTunerDelegator(mainConfig: MainConfig,
                          payload: DataGeneration,
                          testTrainSplitData: Array[TrainSplitReferences])
  extends AbstractGeneticTunerDelegator[DecisionTreeTuner, TreesModelsWithResults, TreesConfig, Any](mainConfig, payload, testTrainSplitData) {


  override protected def initializeTuner: DecisionTreeTuner = {
    val decisionTreeTuner = new DecisionTreeTuner(payload.data, testTrainSplitData, payload.modelType, true)
      .setTreesNumericBoundaries(mainConfig.numericBoundaries)
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
