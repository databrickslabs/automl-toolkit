package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.params.{Defaults, MLPCConfig, MLPCModelsWithResults}
import com.databricks.spark.automatedml.utils.SparkSessionWrapper
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

import org.apache.log4j.{Level, Logger}

class MLPCTuner(df: DataFrame) extends SparkSessionWrapper with Evolution with Defaults {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _scoringMetric = _scoringDefaultClassifier

  private var _mlpcNumericBoundaries = _mlpcDefaultNumBoundaries

  private var _mlpcStringBoundaries = _mlpcDefaultStringBoundaries

  final private val featureInputSize = df.select(_featureCol).head()(0).asInstanceOf[DenseVector].size
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

    @volatile var results = new ArrayBuffer[MLPCModelsWithResults]
    @volatile var modelCnt = 0
    val runs = battery.par
    runs.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(_parallelism))

    val currentStatus = f"Starting Generation $generation \n\t\t Completion Status: ${
      calculateModelingFamilyRemainingTime(generation, modelCnt)}%2.4f%%"

    println(currentStatus)
    logger.log(Level.INFO, currentStatus)

    runs.foreach { x =>
      val runId = java.util.UUID.randomUUID()
      println(s"Starting run $runId with Params: ${x.toString}")

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
      modelCnt += 1
      val runScoreStatement = s"\tFinished run $runId with score: ${scores.sum / scores.length}"
      val progressStatement = f"\t\t Current modeling progress complete in family: ${
        calculateModelingFamilyRemainingTime(generation, modelCnt)}%2.4f%%"
      println(runScoreStatement)
      println(progressStatement)
      logger.log(Level.INFO, runScoreStatement)
      logger.log(Level.INFO, progressStatement)
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


