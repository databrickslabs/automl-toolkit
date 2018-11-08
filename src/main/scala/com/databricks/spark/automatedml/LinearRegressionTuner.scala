package com.databricks.spark.automatedml

import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class LinearRegressionTuner(df: DataFrame) extends SparkSessionWrapper
  with Evolution {

  private var _scoringMetric = "rmse"
  private var _linearRegressionNumericBoundaries = Map(
    "elasticNetParams" -> Tuple2(0.0, 1.0),
    "maxIter" -> Tuple2(100, 10000),
    "regParam" -> Tuple2(0.0, 1.0),
    "tol" -> Tuple2(1E-9, 1E-5)
  )
  private var _linearRegressionStringBoundaries = Map(
    "loss" -> List("squaredError", "huber")
  )

  def setScoringMetric(value: String): this.type = {
    require(regressionMetrics.contains(value),
      s"Regressor scoring optimization '$value' is not a valid member of ${
        invalidateSelection(value, regressionMetrics)
      }")
    this._scoringMetric = value
    this
  }

  def setLinearRegressionNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    this._linearRegressionNumericBoundaries = value
    this
  }

  def setLinearRegressionStringBoundaries(value: Map[String, List[String]]): this.type = {
    this._linearRegressionStringBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getLinearRegressionNumericBoundaries: Map[String, (Double, Double)] = _linearRegressionNumericBoundaries

  def getLinearRegressionStringBoundaries: Map[String, List[String]] = _linearRegressionStringBoundaries

  private def configureModel(modelConfig: LinearRegressionConfig): LinearRegression = {
    new LinearRegression()
      .setLabelCol(_labelCol)
      .setFeaturesCol(_featureCol)
      .setElasticNetParam(modelConfig.elasticNetParam)
      .setFitIntercept(modelConfig.fitIntercept)
      .setLoss(modelConfig.loss)
      .setMaxIter(modelConfig.maxIter)
      .setRegParam(modelConfig.regParam)
      .setSolver("auto")
      .setStandardization(modelConfig.standardization)
      .setTol(modelConfig.tolerance)
  }

  private def generateThresholdedParams(iterationCount: Int): Array[LinearRegressionConfig] = {

    val iterations = new ArrayBuffer[LinearRegressionConfig]

    var i = 0
    do {
      val elasticNetParams = generateRandomDouble("elasticNetParams", _linearRegressionNumericBoundaries)
      val fitIntercept = coinFlip()
      val loss = generateRandomString("loss", _linearRegressionStringBoundaries)
      val maxIter = generateRandomInteger("maxIter", _linearRegressionNumericBoundaries)
      val regParam = generateRandomDouble("regParam", _linearRegressionNumericBoundaries)
      val standardization = coinFlip()
      val tol = generateRandomDouble("tolerance", _linearRegressionNumericBoundaries)
      iterations += LinearRegressionConfig(elasticNetParams, fitIntercept, loss, maxIter, regParam, standardization,
        tol)
      i += 1
    } while (i < iterationCount)

    iterations.toArray
  }

  private def generateAndScoreLinearRegression(train: DataFrame, test: DataFrame,
                                               modelConfig: LinearRegressionConfig, generation: Int)

  // params:
  // setElasticNetParams: Double (0, 1) +
  // setFeaturesCol +
  // setFitIntercept (BOOL) +
  // setLabelCol (string) +
  // setLoss (string: squaredError, huber) -
  // setMaxIter Int (100 -> ?) +
  // setPredictionCol (string) +
  // setRegParam (Double 0 - 1?) +
  // setSolver(string: l-bfgs, normal, auto (default) - should leave on auto since huber solver will throw exceptions if set to 'normal')
  // setStandardization (BOOL) +
  // setTol (smaller = more iterations default 1e-6) +





}
