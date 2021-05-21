package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.AbstractGeneticTunerDelegator
import com.databricks.labs.automl.model.MLPCTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel

import scala.collection.mutable.ArrayBuffer

private[tuner] class MlpcTunerDelegator(mainConfig: MainConfig,
                         payload: DataGeneration,
                         testTrainSplitData: Array[TrainSplitReferences])
  extends AbstractGeneticTunerDelegator
    [MLPCTuner,
      MLPCModelsWithResults,
      MLPCConfig,
      MultilayerPerceptronClassificationModel](mainConfig, payload, testTrainSplitData) {

  override protected def initializeTuner: MLPCTuner = {
    val mlpcTuner = new MLPCTuner(payload.data, testTrainSplitData, true)
      .setMlpcNumericBoundaries(numericBoundaries.get)
      .setMlpcStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(mlpcTuner)
    mlpcTuner
  }

  override protected def delegateTuning: TunerOutput = {

    payload.modelType match {
      case "classifier" => {
        super.delegateTuning
      }

      case _ =>
        throw new UnsupportedOperationException(
          s"Detected Model Type ${payload.modelType} is not supported by MultiLayer Perceptron Classifier"
        )
    }
  }


  override protected def modelOptimization(tuner: MLPCTuner,
                                           genericResults: ArrayBuffer[GenericModelReturn]): Array[MLPCConfig] = {
    postModelingOptimization("MLPC")
      .setNumericBoundaries(tuner.getMlpcNumericBoundaries)
      .setStringBoundaries(tuner.getMlpcStringBoundaries)
      .mlpcPrediction(
        genericResults.result.toArray,
        mainConfig.geneticConfig.hyperSpaceModelType,
        mainConfig.geneticConfig.hyperSpaceModelCount,
        tuner.getFeatureInputSize,
        tuner.getClassDistinctCount
      )
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }
}

object MlpcTunerDelegator {
  def apply(mainConfig: MainConfig, payload: DataGeneration, testTrainSplitData: Array[TrainSplitReferences]):
    MlpcTunerDelegator = new MlpcTunerDelegator(mainConfig, payload, testTrainSplitData)
}
