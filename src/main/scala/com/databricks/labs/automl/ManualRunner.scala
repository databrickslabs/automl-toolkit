package com.databricks.labs.automl

import com.databricks.labs.automl.params._
import com.databricks.labs.automl.reports.{
  DecisionTreeSplits,
  RandomForestFeatureImportance
}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, count}

class ManualRunner(dataPayload: DataGeneration)
    extends AutomationRunner(dataPayload.data) {

  override def exploreFeatureImportances(): FeatureImportanceReturn = {
    val featureResults = new RandomForestFeatureImportance(
      dataPayload.data,
      _featureImportancesConfig,
      dataPayload.modelType
    ).setCutoffType(_mainConfig.featureImportanceCutoffType)
      .setCutoffValue(_mainConfig.featureImportanceCutoffValue)
      .runFeatureImportances(dataPayload.fields)

    params.FeatureImportanceReturn(
      featureResults._1,
      featureResults._2,
      featureResults._3,
      dataPayload.modelType
    )
  }

  override def run(): AutomationOutput = {

    val tunerResult = executeTuning(dataPayload)

    new AutomationOutput(mlFlowOutput = tunerResult.mlFlowOutput) {
      override def modelReport: Array[GenericModelReturn] =
        tunerResult.modelReport
      override def generationReport: Array[GenerationalReport] =
        tunerResult.generationReport
      override def modelReportDataFrame: DataFrame =
        tunerResult.modelReportDataFrame
      override def generationReportDataFrame: DataFrame =
        tunerResult.generationReportDataFrame
    }

  }

  override def generateDecisionSplits(): TreeSplitReport = {

    new DecisionTreeSplits(
      dataPayload.data,
      _treeSplitsConfig,
      dataPayload.modelType
    ).runTreeSplitAnalysis(dataPayload.fields)

  }

  override def runWithFeatureCulling(): FeatureImportanceOutput = {

    val featureImportanceResults = exploreFeatureImportances()
    val selectableFields = featureImportanceResults.fields :+ _mainConfig.labelCol

    val dataSubset = dataPayload.data.select(selectableFields.map(col): _*)
    val runResults =
      new AutomationRunner(dataSubset).setMainConfig(_mainConfig).run()

    new FeatureImportanceOutput(
      featureImportanceResults.data,
      mlFlowOutput = runResults.mlFlowOutput
    ) {
      override def modelReport: Array[GenericModelReturn] =
        runResults.modelReport
      override def generationReport: Array[GenerationalReport] =
        runResults.generationReport
      override def modelReportDataFrame: DataFrame =
        runResults.modelReportDataFrame
      override def generationReportDataFrame: DataFrame =
        runResults.generationReportDataFrame
    }
  }

  override def runFeatureCullingWithPrediction()
    : FeatureImportancePredictionOutput = {

    val featureImportanceResults = exploreFeatureImportances()
    val selectableFields = featureImportanceResults.fields :+ _mainConfig.labelCol

    val dataSubset = dataPayload.data.select(selectableFields.map(col): _*)
    val payload =
      DataGeneration(dataSubset, selectableFields, dataPayload.modelType)

    val runResults = new AutomationRunner(dataSubset)
      .setMainConfig(_mainConfig)
      .executeTuning(payload)
    val predictedData = predictFromBestModel(
      runResults.modelReport,
      runResults.rawData,
      runResults.modelSelection
    )

    new FeatureImportancePredictionOutput(
      featureImportances = featureImportanceResults.data,
      predictionData = predictedData,
      mlFlowOutput = runResults.mlFlowOutput
    ) {
      override def modelReport: Array[GenericModelReturn] =
        runResults.modelReport
      override def generationReport: Array[GenerationalReport] =
        runResults.generationReport
      override def modelReportDataFrame: DataFrame =
        runResults.modelReportDataFrame
      override def generationReportDataFrame: DataFrame =
        runResults.generationReportDataFrame
    }
  }

  override def runWithPrediction(): PredictionOutput = {

    val tunerResult = executeTuning(dataPayload)

    val predictedData = predictFromBestModel(
      tunerResult.modelReport,
      tunerResult.rawData,
      tunerResult.modelSelection
    )

    new PredictionOutput(
      dataWithPredictions = predictedData,
      mlFlowOutput = tunerResult.mlFlowOutput
    ) {
      override def modelReport: Array[GenericModelReturn] =
        tunerResult.modelReport
      override def generationReport: Array[GenerationalReport] =
        tunerResult.generationReport
      override def modelReportDataFrame: DataFrame =
        tunerResult.modelReportDataFrame
      override def generationReportDataFrame: DataFrame =
        tunerResult.generationReportDataFrame
    }

  }

  override def runWithConfusionReport(): ConfusionOutput = {

    val predictionPayload = runWithPrediction()

    val confusionData = predictionPayload.dataWithPredictions
      .select("prediction", _labelCol)
      .groupBy("prediction", _labelCol)
      .agg(count("*").alias("count"))

    new ConfusionOutput(
      predictionData = predictionPayload.dataWithPredictions,
      confusionData = confusionData,
      mlFlowOutput = predictionPayload.mlFlowOutput
    ) {
      override def modelReport: Array[GenericModelReturn] =
        predictionPayload.modelReport
      override def generationReport: Array[GenerationalReport] =
        predictionPayload.generationReport
      override def modelReportDataFrame: DataFrame =
        predictionPayload.modelReportDataFrame
      override def generationReportDataFrame: DataFrame =
        predictionPayload.generationReportDataFrame
    }

  }

}
