package com.databricks.spark.automatedml

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer


class MLPCTuner(df: DataFrame) extends SparkSessionWrapper with Evolution {

  private var _scoringMetric = "f1"

  private var _mlpcNumericBoundaries = Map(
    "layers" -> Tuple2(1.0, 10.0),
    "maxIter" -> Tuple2(100.0, 10000.0),
    "stepSize" -> Tuple2(0.01, 1.0),
    "tol" -> Tuple2(1E-9, 1E-5),
    "hiddenLayerSizeAdjust" -> Tuple2(0.0, 50.0)
  )

  private var _mlpcStringBoundaries = Map(
    "solver" -> List("gd", "l-bfgs")
  )

  final private val featureInputSize = df.select(_featureCol).head()(0).asInstanceOf[SparseVector].size
  final private val classDistinctCount = df.select(_labelCol).distinct().count().toInt


  def setScoringMetric(value: String): this.type = {
    require(classificationMetrics.contains(value),
      s"Classification scoring metric $value is not a valid member of ${
        invalidateSelection(value, classificationMetrics)}")
    _scoringMetric = value
    this
  }

  def setMlpcNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    _mlpcNumericBoundaries = value
    this
  }

  def setMlpcStringBoundaries(value: Map[String, List[String]]): this.type = {
    _mlpcStringBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getMlpcNumericBoundaries: Map[String, (Double, Double)] = _mlpcNumericBoundaries

  def getMlpcStringBoundaries: Map[String, List[String]] = _mlpcStringBoundaries

  private def configureModel(modelConfig: MLPCConfig): MultilayerPerceptronClassifier = {
    new MultilayerPerceptronClassifier()
      .setLabelCol(_labelCol)
      .setFeaturesCol(_featureCol)
      .setLayers(modelConfig.layers)
      .setMaxIter(modelConfig.maxIter)
      .setSolver(modelConfig.solver)
      .setStepSize(modelConfig.stepSize)
      .setTol(modelConfig.tol)
  }

  private def generateThresholdedParams(iterationCount: Int): Array[MLPCConfig] = {

    val iterations = new ArrayBuffer[MLPCConfig]

    var i = 0
    do {
      val layers = generateLayerArray("layers", "hiddenLayerSizeAdjust",
        _mlpcNumericBoundaries, featureInputSize, classDistinctCount)
      val maxIter = generateRandomInteger("maxIter", _mlpcNumericBoundaries)
      val solver = generateRandomString("solver", _mlpcStringBoundaries)
      val stepSize = generateRandomDouble("stepSize", _mlpcNumericBoundaries)
      val tol = generateRandomDouble("tol", _mlpcNumericBoundaries)
      iterations += MLPCConfig(layers, maxIter, solver, stepSize, tol)
      i += 1
    } while (i < iterationCount)
    iterations.toArray
  }

  private def generateAndScoreMLPCModel(train: DataFrame, test: DataFrame,
                                        modelConfig: MLPCConfig,
                                        generation: Int = 1): MLPCModelsWithResults = {

    val mlpcModel = configureModel(modelConfig)
    val builtModel = mlpcModel.fit(train)
    val predictedData = builtModel.transform(test)
    val scoringMap = scala.collection.mutable.Map[String, Double]()

    for(i <- classificationMetrics){
      val scoreEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol(_labelCol)
        .setPredictionCol("prediction")
        .setMetricName(i)
      scoringMap(i) = scoreEvaluator.evaluate(predictedData)
    }

    MLPCModelsWithResults(modelConfig, builtModel, scoringMap(_scoringMetric), scoringMap.toMap, generation)

  }

//TODO: resume here.


}


