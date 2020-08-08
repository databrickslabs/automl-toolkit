package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, MainConfig, TunerOutput}
object TuningRunner {

  def runTuning(mainConfig: MainConfig,
                payload: DataGeneration,
                testTrainSplitData: Array[TrainSplitReferences]): TunerOutput = {




//    val generationalData = extractGenerationalScores(
//      genericResultData,
//      _mainConfig.scoringOptimizationStrategy,
//      _mainConfig.modelFamily,
//      modelSelection
//    )

    null
  }

}
