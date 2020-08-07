package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.ensemble.tuner.validate.GeneticTunerValidator
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, GenerationalReport, GenericModelReturn, MainConfig, RandomForestModelsWithResults, TunerOutput}
import com.databricks.labs.automl.utils.AutomationTools
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

trait GeneticTuner extends GeneticTunerValidator  with AutomationTools {

  protected def delegateTuning(mainConfig: MainConfig,
                               payload: DataGeneration,
                               testTrainSplitData: Array[TrainSplitReferences]): TunerOutput

  def tune(mainConfig: MainConfig,
           payload: DataGeneration,
           testTrainSplitData: Array[TrainSplitReferences]): TunerOutput = {
    validate(mainConfig)
    delegateTuning(mainConfig, payload, testTrainSplitData)
  }

  def tunerOutput(mainConfig: MainConfig,
                  results: Array[RandomForestModelsWithResults],
                  modelStats: DataFrame,
                  modelSelection: String,
                  dataframe: DataFrame): TunerOutput = {

    val genericResults = new ArrayBuffer[GenericModelReturn]

    results.foreach { x =>
      genericResults += GenericModelReturn(
        hyperParams = extractPayload(x.modelHyperParams),
        model = x.model,
        score = x.score,
        metrics = x.evalMetrics,
        generation = x.generation
      )
    }

    val genericResultData = genericResults.result.toArray

    val generationalData = extractGenerationalScores(
      genericResultData,
      mainConfig.scoringOptimizationStrategy,
      mainConfig.modelFamily,
      modelSelection
    )

    new TunerOutput(
      rawData = dataframe,
      modelSelection = modelSelection,
      mlFlowOutput = mlFlow
    ) {
      override def modelReport: Array[GenericModelReturn] = genericResultData
      override def generationReport: Array[GenerationalReport] =
        generationalData
      override def modelReportDataFrame: DataFrame = modelStats
      override def generationReportDataFrame: DataFrame =
        generationDataFrameReport(
          generationalData,
          mainConfig.scoringOptimizationStrategy
        )
    }
  }

}
