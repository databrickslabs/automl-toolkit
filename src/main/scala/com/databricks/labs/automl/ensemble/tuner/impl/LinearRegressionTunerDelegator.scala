package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.AbstractGeneticTunerDelegator
import com.databricks.labs.automl.model.LinearRegressionTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._
import org.apache.spark.ml.regression.LinearRegressionModel

import scala.collection.mutable.ArrayBuffer

private[tuner] class LinearRegressionTunerDelegator(mainConfig: MainConfig,
                                     payload: DataGeneration,
                                     testTrainSplitData: Array[TrainSplitReferences])
  extends AbstractGeneticTunerDelegator
     [LinearRegressionTuner,
      LinearRegressionModelsWithResults,
      LinearRegressionConfig,
      LinearRegressionModel](mainConfig, payload, testTrainSplitData) {

  override protected def initializeTuner: LinearRegressionTuner = {
    val linearRegressionTuner = new LinearRegressionTuner(
      payload.data,
      testTrainSplitData,
      true
    ).setLinearRegressionNumericBoundaries(numericBoundaries.get)
      .setLinearRegressionStringBoundaries(mainConfig.stringBoundaries)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(linearRegressionTuner)
    linearRegressionTuner
  }

  override protected def delegateTuning: TunerOutput = {

    payload.modelType match {
      case "regressor" => {
        super.delegateTuning
      }

      case _ =>
        throw new UnsupportedOperationException(
          s"Detected Model Type ${payload.modelType} is not supported by Linear Regression"
        )
    }
  }

  override protected def modelOptimization(tuner: LinearRegressionTuner,
                                           genericResults: ArrayBuffer[GenericModelReturn]): Array[LinearRegressionConfig] = {
    postModelingOptimization("LinearRegression")
      .setNumericBoundaries(tuner.getLinearRegressionNumericBoundaries)
      .setStringBoundaries(tuner.getLinearRegressionStringBoundaries)
      .linearRegressionPrediction(
        genericResults.result.toArray,
        mainConfig.geneticConfig.hyperSpaceModelType,
        mainConfig.geneticConfig.hyperSpaceModelCount
      )
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }
}

object LinearRegressionTunerDelegator {
  def apply(mainConfig: MainConfig, payload: DataGeneration, testTrainSplitData: Array[TrainSplitReferences]):
    LinearRegressionTunerDelegator = new LinearRegressionTunerDelegator(mainConfig, payload, testTrainSplitData)
}