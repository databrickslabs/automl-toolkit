package com.databricks.spark.automatedml

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

import scala.collection.mutable.{ArrayBuffer, ListBuffer}


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
  
  private def runBattery(battery: Array[MLPCConfig], generation: Int = 1): Array[MLPCModelsWithResults] = {

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    val results = new ArrayBuffer[MLPCModelsWithResults]
    val runs = battery.par

    runs.foreach { x =>

      val kFoldBuffer = new ArrayBuffer[MLPCModelsWithResults]

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) = genTestTrain(df, scala.util.Random.nextLong)
        kFoldBuffer += generateAndScoreMLPCModel(train, test, x)
      }
      val scores = new ArrayBuffer[Double]
      kFoldBuffer.map(x => {
        scores += x.score
      })

      val scoringMap = scala.collection.mutable.Map[String, Double]()
      for (a <- classificationMetrics) {
        val metricScores = new ListBuffer[Double]
        kFoldBuffer.map(x => metricScores += x.evalMetrics(a))
        scoringMap(a) = metricScores.sum / metricScores.length
      }
      val runAvg = MLPCModelsWithResults(x, kFoldBuffer.result.head.model, scores.sum / scores.length,
        scoringMap.toMap, generation)
      results += runAvg

    }
    _optimizationStrategy match {
      case "minimize" => results.toArray.sortWith(_.score < _.score)
      case _ => results.toArray.sortWith(_.score > _.score)
    }

  }

  private def irradiateGeneration(parents: Array[MLPCConfig], mutationCount: Int,
                                  mutationAggression: Int, mutationMagnitude: Double): Array[MLPCConfig] = {

    val mutationPayload = new ArrayBuffer[MLPCConfig]
    val totalConfigs = modelConfigLength[MLPCConfig]
    val indexMutation = if (mutationAggression >= totalConfigs) totalConfigs - 1 else totalConfigs - mutationAggression
    val mutationCandidates = generateThresholdedParams(mutationCount)
    val mutationIndeces = generateMutationIndeces(1, totalConfigs, indexMutation, mutationCount)

    for (i <- mutationCandidates.indices) {

      val randomParent = scala.util.Random.shuffle(parents.toList).head
      val mutationIteration = mutationCandidates(i)
      val mutationIndexIteration = mutationIndeces(i)

      mutationPayload += MLPCConfig(
        if (mutationIndexIteration.contains(0)) geneMixing(randomParent.layers,
          mutationIteration.layers, mutationMagnitude)
        else randomParent.layers,
        if (mutationIndexIteration.contains(1)) geneMixing(randomParent.maxIter,
          mutationIteration.maxIter, mutationMagnitude)
        else randomParent.maxIter,
        if (mutationIndexIteration.contains(2)) geneMixing(randomParent.solver,
          mutationIteration.solver)
        else randomParent.solver,
        if (mutationIndexIteration.contains(3)) geneMixing(randomParent.stepSize,
          mutationIteration.stepSize, mutationMagnitude)
        else randomParent.stepSize,
        if (mutationIndexIteration.contains(4)) geneMixing(randomParent.tol,
          mutationIteration.tol, mutationMagnitude)
        else randomParent.tol
      )
    }
    mutationPayload.result.toArray
  }

  def generateIdealParents(results: Array[MLPCModelsWithResults]): Array[MLPCConfig] = {
    val bestParents = new ArrayBuffer[MLPCConfig]
    results.take(_numberOfParentsToRetain).map(x => {
      bestParents += x.modelHyperParams
    })
    bestParents.result.toArray
  }

  def evolveParameters(startingSeed: Option[MLPCConfig] = None): Array[MLPCModelsWithResults] = {

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[MLPCModelsWithResults]

    val totalConfigs = modelConfigLength[MLPCConfig]

    val primordial = startingSeed match {
      case Some(`startingSeed`) =>
        val generativeArray = new ArrayBuffer[MLPCConfig]
        generativeArray += startingSeed.asInstanceOf[MLPCConfig]
        generativeArray ++= irradiateGeneration(
          Array(startingSeed.asInstanceOf[MLPCConfig]),
          _firstGenerationGenePool, totalConfigs - 1, _geneticMixing)
        runBattery(generativeArray.result.toArray, generation)
      case _ => runBattery(generateThresholdedParams(_firstGenerationGenePool), generation)
    }

    fossilRecord ++= primordial
    generation += 1

    (1 to _numberOfMutationGenerations).map(i => {

      val mutationAggressiveness = _generationalMutationStrategy match {
        case "linear" => if (totalConfigs - (i + 1) < 1) 1 else totalConfigs - (i + 1)
        case _ => _fixedMutationValue
      }

      val currentState = _optimizationStrategy match {
        case "minimize" => fossilRecord.result.toArray.sortWith(_.score < _.score)
        case _ => fossilRecord.result.toArray.sortWith(_.score > _.score)
      }

      val evolution = irradiateGeneration(generateIdealParents(currentState), _numberOfMutationsPerGeneration,
        mutationAggressiveness, _geneticMixing)

      var evolve = runBattery(evolution, generation)
      generation += 1
      fossilRecord ++= evolve

    })

    _optimizationStrategy match {
      case "minimize" => fossilRecord.result.toArray.sortWith(_.score < _.score)
      case _ => fossilRecord.result.toArray.sortWith(_.score > _.score)
    }
  }

  def evolveBest(startingSeed: Option[MLPCConfig] = None): MLPCModelsWithResults = {
    evolveParameters(startingSeed).head
  }

  def generateScoredDataFrame(results: Array[MLPCModelsWithResults]): DataFrame = {

    import spark.sqlContext.implicits._

    val scoreBuffer = new ListBuffer[(Int, Double)]
    results.map(x => scoreBuffer += ((x.generation, x.score)))
    val scored = scoreBuffer.result
    spark.sparkContext.parallelize(scored)
      .toDF("generation", "score").orderBy(col("generation").asc, col("score").asc)
  }

  def evolveWithScoringDF(startingSeed: Option[MLPCConfig] = None):
  (Array[MLPCModelsWithResults], DataFrame) = {
    val evolutionResults = evolveParameters(startingSeed)
    (evolutionResults, generateScoredDataFrame(evolutionResults))
  }


}


