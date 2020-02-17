package com.databricks.labs.automl.model

import com.databricks.labs.automl.params.{
  Defaults,
  NaiveBayesConfig,
  NaiveBayesModelsWithResults
}
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.log4j.Logger
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer

class NaiveBayesTuner(df: DataFrame)
    extends SparkSessionWrapper
    with Defaults
    with Evolution {

  //TODO: finish this some time.

  // Perform a check to validate the structure and conditions of the input DataFrame to ensure that it can be modeled
  validateInputDataframe(df)

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _scoringMetric = _scoringDefaultClassifier

  private var _naiveBayesNumericBoundaries = _naiveBayesDefaultNumBoundaries

  private var _naiveBayesStringBoundaries = _naiveBayesDefaultStringBoundaries

  private var _classificationMetrics = classificationMetrics

  private var _naiveBayesThresholds = calculateThresholds()

  def setScoringMetric(value: String): this.type = {
    require(
      classificationMetrics.contains(value),
      s"Classification scoring metric $value is not a valid member of ${invalidateSelection(value, classificationMetrics)}"
    )
    this._scoringMetric = value
    this
  }

  def setNaiveBayesNumericBoundaries(
    value: Map[String, (Double, Double)]
  ): this.type = {
    this._naiveBayesNumericBoundaries = value
    this
  }

  def setNaiveBayesStringBoundaries(
    value: Map[String, List[String]]
  ): this.type = {
    this._naiveBayesStringBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getNaiveBayesNumericBoundaries: Map[String, (Double, Double)] =
    _naiveBayesNumericBoundaries

  def getNaiveBayesStringBoundaries: Map[String, List[String]] =
    _naiveBayesStringBoundaries

  def getClassificationMetrics: List[String] = classificationMetrics

  private def resetClassificationMetrics: List[String] =
    classificationMetricValidator(
      classificationAdjudicator(df),
      classificationMetrics
    )

  private def setClassificationMetrics(value: List[String]): this.type = {
    _classificationMetrics = value
    this
  }

  private def calculateThresholds(): Array[Double] = {

    val uniqueLabels = df
      .select(_labelCol)
      .groupBy(col(_labelCol))
      .agg(count("*"))
      .alias("counts")
      .orderBy(col("counts").desc)
      .collect()

    val values = uniqueLabels.map(x => x.getAs[Double]("counts"))

    val totals = values.sum

    values.map(x => x / totals)

  }

  private def configureModel(modelConfig: NaiveBayesConfig): NaiveBayes = {

    val nbModel = new NaiveBayes()
      .setFeaturesCol(_featureCol)
      .setLabelCol(_labelCol)
      .setSmoothing(modelConfig.smoothing)

    if (modelConfig.thresholds) nbModel.setThresholds(_naiveBayesThresholds)

    nbModel
  }

  private def returnBestHyperParameters(
    collection: ArrayBuffer[NaiveBayesModelsWithResults]
  ): (NaiveBayesConfig, Double) = {

    val bestEntry = _optimizationStrategy match {
      case "minimize" =>
        collection.result.toArray.sortWith(_.score < _.score).head
      case _ => collection.result.toArray.sortWith(_.score > _.score).head
    }
    (bestEntry.modelHyperParams, bestEntry.score)

  }

  private def evaluateStoppingScore(currentBestScore: Double,
                                    stopThreshold: Double): Boolean = {
    _optimizationStrategy match {
      case "minimize" => if (currentBestScore > stopThreshold) true else false
      case _          => if (currentBestScore < stopThreshold) true else false
    }
  }

  private def evaluateBestScore(runScore: Double,
                                bestScore: Double): Boolean = {
    _optimizationStrategy match {
      case "minimize" => if (runScore < bestScore) true else false
      case _          => if (runScore > bestScore) true else false
    }
  }

  private def sortAndReturnAll(
    results: ArrayBuffer[NaiveBayesModelsWithResults]
  ): Array[NaiveBayesModelsWithResults] = {
    _optimizationStrategy match {
      case "minimize" => results.result.toArray.sortWith(_.score < _.score)
      case _          => results.result.toArray.sortWith(_.score > _.score)
    }
  }

  private def sortAndReturnBestScore(
    results: ArrayBuffer[NaiveBayesModelsWithResults]
  ): Double = {
    sortAndReturnAll(results).head.score
  }

  private def generateThresholdedParams(
    iterationCount: Int
  ): Array[NaiveBayesConfig] = {

    val iterations = new ArrayBuffer[NaiveBayesConfig]

    var i = 0
    do {
      val modelType =
        generateRandomString("modelType", _naiveBayesStringBoundaries)
      val smoothing =
        generateRandomDouble("smoothing", _naiveBayesNumericBoundaries)
      val thresholds = coinFlip()
      iterations += NaiveBayesConfig(modelType, smoothing, thresholds)
      i += 1
    } while (i < iterationCount)
    iterations.toArray
  }

  private def generateAndScoreNaiveBayes(
    train: DataFrame,
    test: DataFrame,
    modelConfig: NaiveBayesConfig,
    generation: Int = 1
  ): NaiveBayesModelsWithResults = {
    val model = configureModel(modelConfig)

    val builtModel = model.fit(train)

    val predictedData = builtModel.transform(test)

    val scoringMap = scala.collection.mutable.Map[String, Double]()

    for (i <- _classificationMetrics) {
      scoringMap(i) = classificationScoring(i, _labelCol, predictedData)
    }
    NaiveBayesModelsWithResults(
      modelConfig,
      builtModel,
      scoringMap(_scoringMetric),
      scoringMap.toMap,
      generation
    )
  }

}
