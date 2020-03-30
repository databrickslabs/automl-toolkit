package com.databricks.labs.automl

class AutomationRunnerIT extends AbstractUnitSpec {

  it should "return confusion report for Logistic Regression in batch evolution strategy" in {
    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
    val fullConfig = AutomationUnitTestsUtil.getLogisticRegressionConfig(
      adultDfwithLabel,
      "batch"
    )
    val confusionOutput = fullConfig.runWithConfusionReport()
    AutomationUnitTestsUtil.assertConfusionOutput(confusionOutput)
    AutomationUnitTestsUtil.assertPredOutput(
      adultDfwithLabel.count(),
      confusionOutput.predictionData.count()
    )
  }

  it should "return confusion report for Random Forest in batch evolution strategy" in {
    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
    val fullConfig =
      AutomationUnitTestsUtil.getRandomForestConfig(adultDfwithLabel, "batch")
    val confusionOutput = fullConfig.runWithConfusionReport()
    AutomationUnitTestsUtil.assertConfusionOutput(confusionOutput)
    AutomationUnitTestsUtil.assertPredOutput(
      adultDfwithLabel.count(),
      confusionOutput.predictionData.count()
    )
  }

  it should "return confusion report for XgBoost in batch evolution strategy" in {
    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
    adultDfwithLabel.show(1000)
    val fullConfig =
      AutomationUnitTestsUtil.getXgBoostConfig(adultDfwithLabel, "batch")
    val confusionOutput = fullConfig.runWithConfusionReport()
    AutomationUnitTestsUtil.assertConfusionOutput(confusionOutput)
    AutomationUnitTestsUtil.assertPredOutput(
      adultDfwithLabel.count(),
      confusionOutput.predictionData.count()
    )
  }

  it should "return confusion report for Logistic Regression in continuous evolution strategy" in {
    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
    val fullConfig = AutomationUnitTestsUtil.getLogisticRegressionConfig(
      adultDfwithLabel,
      "continuous"
    )
    val confusionOutput = fullConfig.runWithConfusionReport()
    AutomationUnitTestsUtil.assertConfusionOutput(confusionOutput)
    AutomationUnitTestsUtil.assertPredOutput(
      adultDfwithLabel.count(),
      confusionOutput.predictionData.count()
    )
  }

  it should "return confusion report for Random Forest in continuous evolution strategy" in {
    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
    val fullConfig = AutomationUnitTestsUtil.getRandomForestConfig(
      adultDfwithLabel,
      "continuous"
    )
    val confusionOutput = fullConfig.runWithConfusionReport()
    AutomationUnitTestsUtil.assertConfusionOutput(confusionOutput)
    AutomationUnitTestsUtil.assertPredOutput(
      adultDfwithLabel.count(),
      confusionOutput.predictionData.count()
    )
  }

  it should "return confusion report for XgBoost in continuous evolution strategy" in {
    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
    val fullConfig =
      AutomationUnitTestsUtil.getXgBoostConfig(adultDfwithLabel, "continuous")
    val confusionOutput = fullConfig.runWithConfusionReport()
    AutomationUnitTestsUtil.assertConfusionOutput(confusionOutput)
    AutomationUnitTestsUtil.assertPredOutput(
      adultDfwithLabel.count(),
      confusionOutput.predictionData.count()
    )
  }

  it should "return predictions with XgBoost" in {
    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
    val fullConfig =
      AutomationUnitTestsUtil.getXgBoostConfig(adultDfwithLabel, "continuous")
    val predictionRowsCount =
      fullConfig.runWithPrediction().dataWithPredictions.count()
    assert(
      predictionRowsCount > 0,
      "should have returned more than 0 rows for prediction output"
    )
    assert(
      predictionRowsCount == adultDfwithLabel.count(),
      "should have same row count for input dataset and prediction dataset"
    )
  }

  it should "return predictions with XgBoost when feature interaction mode is on" in {
    val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
    val fullConfig = AutomationUnitTestsUtil
      .getXgBoostConfig(adultDfwithLabel, "continuous")
      .featureInteractionOn()
      .setFeatureInteractionRetentionMode("all")
      .setFeatureInteractionParallelism(2)
    val confusionOutput = fullConfig.runWithConfusionReport()
    AutomationUnitTestsUtil.assertConfusionOutput(confusionOutput)
    AutomationUnitTestsUtil.assertPredOutput(
      adultDfwithLabel.count(),
      confusionOutput.predictionData.count()
    )
  }

}
