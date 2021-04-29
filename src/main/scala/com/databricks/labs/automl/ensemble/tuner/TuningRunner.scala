package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, MainConfig, TunerOutput}

private[ensemble] object TuningRunner {

  def runTuning(mainConfig: MainConfig,
                payload: DataGeneration,
                testTrainSplitData: Array[TrainSplitReferences]): TunerOutput = {
    val tuner = ModelFamilyTunerType.getTunerInstanceByModelFamily(
      mainConfig.modelFamily,
      mainConfig,
      payload,
      testTrainSplitData
    )
    tuner.tune
  }



}
