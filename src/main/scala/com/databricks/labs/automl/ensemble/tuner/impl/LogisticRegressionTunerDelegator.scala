package com.databricks.labs.automl.ensemble.tuner.impl

import com.databricks.labs.automl.ensemble.tuner.AbstractGeneticTunerDelegator
import com.databricks.labs.automl.model.LogisticRegressionTuner
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params._
import org.apache.spark.ml.classification.LogisticRegressionModel

import scala.collection.mutable.ArrayBuffer

private[tuner] class LogisticRegressionTunerDelegator(mainConfig: MainConfig,
                                       payload: DataGeneration,
                                       testTrainSplitData: Array[TrainSplitReferences])
  extends AbstractGeneticTunerDelegator
     [LogisticRegressionTuner,
      LogisticRegressionModelsWithResults,
      LogisticRegressionConfig,
      LogisticRegressionModel](mainConfig, payload, testTrainSplitData) {

  override protected def initializeTuner: LogisticRegressionTuner = {
    val linearRegressionTuner = new LogisticRegressionTuner(payload.data, testTrainSplitData, true)
      .setLogisticRegressionNumericBoundaries(numericBoundaries.get)
      .setScoringMetric(mainConfig.scoringMetric)
    setTunerEvolutionConfig(linearRegressionTuner)
    linearRegressionTuner
  }

  override protected def delegateTuning: TunerOutput = {
    payload.modelType match {
      case "classifier" => {
        super.delegateTuning
      }

      case _ =>
        throw new UnsupportedOperationException(
          s"Detected Model Type ${payload.modelType} is not supported by Logistic Regression"
        )
    }
  }


  override protected def modelOptimization(tuner: LogisticRegressionTuner,
                                             genericResults: ArrayBuffer[GenericModelReturn]): Array[LogisticRegressionConfig] = {
    postModelingOptimization("LogisticRegression")
      .setNumericBoundaries(tuner.getLogisticRegressionNumericBoundaries)
      .setStringBoundaries(mainConfig.stringBoundaries)
      .logisticRegressionPrediction(
        genericResults.result.toArray,
        mainConfig.geneticConfig.hyperSpaceModelType,
        mainConfig.geneticConfig.hyperSpaceModelCount
      )
  }

  override def validate(mainConfig: MainConfig): Unit = {

  }
}

object LogisticRegressionTunerDelegator {
  def apply(mainConfig: MainConfig, payload: DataGeneration, testTrainSplitData: Array[TrainSplitReferences]):
    LogisticRegressionTunerDelegator = new LogisticRegressionTunerDelegator(mainConfig, payload, testTrainSplitData)
}
