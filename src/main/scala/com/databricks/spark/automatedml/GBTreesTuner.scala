package com.databricks.spark.automatedml

import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.DataFrame

class GBTreesTuner(df: DataFrame, modelSelection: String) extends SparkSessionWrapper with Evolution {


  private var _scoringMetric = modelSelection match {
    case "regressor" => "rmse"
    case "classifier" => "f1"
    case _ => throw new UnsupportedOperationException(s"Model $modelSelection is not a supported modeling mode")
  }

  private var _gbtNumericBoundaries = Map(
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "minInstancesPerNode" -> Tuple2(1.0, 50.0),
    "stepSize" -> Tuple2(0.1, 1.0)
  )

  private var _gbtStringBoundaries = Map(
    "impurity" -> List("gini", "entropy"),
    "lossType" -> List("logistic")
  )

  def setScoringMetric(value: String): this.type = {
    modelSelection match {
      case "regressor" => require(regressionMetrics.contains(value),
        s"Regressor scoring metric '$value' is not a valid member of ${
          invalidateSelection(value, regressionMetrics)
        }")
      case "classifier" => require(classificationMetrics.contains(value),
        s"Regressor scoring metric '$value' is not a valid member of ${
          invalidateSelection(value, classificationMetrics)
        }")
      case _ => throw new UnsupportedOperationException(s"Unsupported modelType $modelSelection")
    }
    this._scoringMetric = value
    this
  }

  def setRandomForestNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    _gbtNumericBoundaries = value
    this
  }

  def setRandomForestStringBoundaries(value: Map[String, List[String]]): this.type = {
    _gbtStringBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getRandomForestNumericBoundaries: Map[String, (Double, Double)] = _gbtNumericBoundaries

  def getRandomForestStringBoundaries: Map[String, List[String]] = _gbtStringBoundaries

  def getClassificationMetrics: List[String] = classificationMetrics

  def getRegressionMetrics: List[String] = regressionMetrics

  private def modelDecider[A, B](modelConfig: GBTConfig) = {

    val builtModel = modelSelection match {
      case "classifier" =>
        new GBTClassifier()
          .setLabelCol(_labelCol)
          .setFeaturesCol(_featureCol)
          .setCheckpointInterval(-1)
          .setImpurity(modelConfig.impurity)
          .setLossType(modelConfig.lossType)
          .setMaxBins(modelConfig.maxBins)
          .setMaxDepth(modelConfig.maxDepth)
          .setMaxIter(modelConfig.maxIter)
          .setMinInfoGain(modelConfig.minInfoGain)
          .setMinInstancesPerNode(modelConfig.minInstancesPerNode)
          .setStepSize(modelConfig.stepSize)
      case "regressor" =>
        new GBTRegressor()
          .setLabelCol(_labelCol)
          .setFeaturesCol(_featureCol)
          .setCheckpointInterval(-1)
          .setImpurity(modelConfig.impurity)
          .setLossType(modelConfig.lossType)
          .setMaxBins(modelConfig.maxBins)
          .setMaxDepth(modelConfig.maxDepth)
          .setMaxIter(modelConfig.maxIter)
          .setMinInfoGain(modelConfig.minInfoGain)
          .setMinInstancesPerNode(modelConfig.minInstancesPerNode)
          .setStepSize(modelConfig.stepSize)
      case _ => throw new UnsupportedOperationException(s"Unsupported modelType $modelSelection")
    }
    builtModel
  }

  override def generateRandomString(param: String, boundaryMap: Map[String, List[String]]): String = {

    val stringListing = param match {
      case "impurity" => modelSelection match {
        case "regressor" => List("variance")
        case _ => boundaryMap(param)
      }
      case "lossType" => modelSelection match {
        case "regressor" => List("squared", "absolute")
        case _ => boundaryMap(param)
      }
      case _ => boundaryMap(param)
    }
    _randomizer.shuffle(stringListing).head
  }





}
