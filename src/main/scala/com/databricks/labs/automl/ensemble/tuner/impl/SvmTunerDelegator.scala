package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.AbstractGeneticTunerDelegator
import com.databricks.labs.automl.model.SVMTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._
import org.apache.spark.ml.classification.LinearSVCModel

import scala.collection.mutable.ArrayBuffer

private[tuner] class SvmTunerDelegator(mainConfig: MainConfig,
                        payload: DataGeneration,
                        testTrainSplitData: Array[TrainSplitReferences])
  extends AbstractGeneticTunerDelegator
     [SVMTuner,
      SVMModelsWithResults,
      SVMConfig,
      LinearSVCModel](mainConfig, payload, testTrainSplitData) {

  override protected def initializeTuner: SVMTuner = {
    val svmTuner = new SVMTuner(
      payload.data,
      testTrainSplitData,
      true)
      .setSvmNumericBoundaries(numericBoundaries.get)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(svmTuner)
    svmTuner
  }

  override protected def delegateTuning: TunerOutput = {

    payload.modelType match {
      case "classifier" => {
        super.delegateTuning
      }

      case _ =>
        throw new UnsupportedOperationException(
          s"Detected Model Type ${payload.modelType} is not supported by Support Vector Machines"
        )
    }
  }

  override protected def modelOptimization(tuner: SVMTuner,
                                           genericResults: ArrayBuffer[GenericModelReturn]): Array[SVMConfig] = {
    postModelingOptimization("SVM")
      .setNumericBoundaries(tuner.getSvmNumericBoundaries)
      .setStringBoundaries(mainConfig.stringBoundaries)
      .svmPrediction(
        genericResults.result.toArray,
        mainConfig.geneticConfig.hyperSpaceModelType,
        mainConfig.geneticConfig.hyperSpaceModelCount
      )
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }
}

object SvmTunerDelegator {
  def apply(mainConfig: MainConfig, payload: DataGeneration, testTrainSplitData: Array[TrainSplitReferences]):
    SvmTunerDelegator = new SvmTunerDelegator(mainConfig, payload, testTrainSplitData)
}
