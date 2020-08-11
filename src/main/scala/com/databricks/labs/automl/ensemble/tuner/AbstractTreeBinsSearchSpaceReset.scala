package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.model.AbstractTuner
import com.databricks.labs.automl.model.tools.ModelUtils
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, MainConfig, TunerConfigBase, TunerOutputWithResults}

abstract class AbstractTreeBinsSearchSpaceReset[
  A <: AbstractTuner[C, B, D],
  B <: TunerOutputWithResults[C, D],
  C <: TunerConfigBase,
  D](mainConfig: MainConfig,
    payload: DataGeneration,
    testTrainSplitData: Array[TrainSplitReferences])
  extends
  AbstractGeneticTunerDelegator[A, B, C, D](mainConfig, payload, testTrainSplitData) {

  override def numericBoundaries: Option[Map[String, (Double, Double)]] = {
    Some(ModelUtils.resetTreeBinsSearchSpace(
      payload.data,
      mainConfig.numericBoundaries,
      mainConfig.fieldsToIgnoreInVector,
      mainConfig.labelCol,
      mainConfig.featuresCol
    ))
  }

}
