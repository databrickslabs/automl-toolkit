package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.params.{Defaults, SVMConfig, SVMModelsWithResults}
import com.databricks.spark.automatedml.utils.SparkSessionWrapper
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

import org.apache.log4j.{Level, Logger}

class SVMTuner(df: DataFrame) extends SparkSessionWrapper with Evolution with Defaults {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _scoringMetric = _scoringDefaultRegressor

  private var _svmNumericBoundaries = _svmDefaultNumBoundaries

  def setScoringMetric(value: String): this.type = {
    require(regressionMetrics.contains(value), s"Regressor scoring metric '$value' is not a valid member of ${
      invalidateSelection(value, regressionMetrics)
    }")
    _scoringMetric = value
    this
  }

  def setSvmNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    _svmNumericBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getSvmNumericBoundaries: Map[String, (Double, Double)] = _svmNumericBoundaries

  private def configureModel(modelConfig: SVMConfig): LinearSVC = {
    new LinearSVC()
      .setLabelCol(_labelCol)
      .setFeaturesCol(_featureCol)
      .setFitIntercept(modelConfig.fitIntercept)
      .setMaxIter(modelConfig.maxIter)
      .setRegParam(modelConfig.regParam)
      .setStandardization(modelConfig.standardization)
      .setTol(modelConfig.tol)
  }

  private def generateThresholdedParams(iterationCount: Int): Array[SVMConfig] = {
    val iterations = new ArrayBuffer[SVMConfig]

    var i = 0
    do {
      val fitIntercept = coinFlip()
      val maxIter = generateRandomInteger("maxIter", _svmNumericBoundaries)
      val regParam = generateRandomDouble("regParam", _svmNumericBoundaries)
      val standardization = coinFlip()
      val tol = generateRandomDouble("tol", _svmNumericBoundaries)
      iterations += SVMConfig(fitIntercept, maxIter, regParam, standardization, tol)
    } while (i < iterationCount)
    iterations.toArray
  }

  private def generateAndScoreSVM(train: DataFrame, test: DataFrame,
                                  modelConfig: SVMConfig, generation: Int = 1): SVMModelsWithResults = {

    val svmModel = configureModel(modelConfig)
    val builtModel = svmModel.fit(train)
    val predictedData = builtModel.transform(test)
    val scoringMap = scala.collection.mutable.Map[String, Double]()

    for (i <- regressionMetrics) {
      val scoreEvaluator = new RegressionEvaluator()
        .setLabelCol(_labelCol)
        .setPredictionCol("prediction")
        .setMetricName(i)
      scoringMap(i) = scoreEvaluator.evaluate(predictedData)
    }
    SVMModelsWithResults(modelConfig, builtModel, scoringMap(_scoringMetric),
      scoringMap.toMap, generation)
  }

  private def runBattery(battery: Array[SVMConfig], generation: Int = 1): Array[SVMModelsWithResults] = {

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    @volatile var results = new ArrayBuffer[SVMModelsWithResults]
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

      val kFoldBuffer = new ArrayBuffer[SVMModelsWithResults]

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) = genTestTrain(df, scala.util.Random.nextLong)
        kFoldBuffer += generateAndScoreSVM(train, test, x)
      }
      val scores = new ArrayBuffer[Double]
      kFoldBuffer.map(x => {
        scores += x.score
      })

      val scoringMap = scala.collection.mutable.Map[String, Double]()
      for (a <- regressionMetrics) {
        val metricScores = new ListBuffer[Double]
        kFoldBuffer.map(x => metricScores += x.evalMetrics(a))
        scoringMap(a) = metricScores.sum / metricScores.length
      }
      val runAvg = SVMModelsWithResults(x, kFoldBuffer.result.head.model, scores.sum / scores.length,
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

  private def irradiateGeneration(parents: Array[SVMConfig], mutationCount: Int,
                                  mutationAggression: Int, mutationMagnitude: Double): Array[SVMConfig] = {

    val mutationPayload = new ArrayBuffer[SVMConfig]
    val totalConfigs = modelConfigLength[SVMConfig]
    val indexMutation = if (mutationAggression >= totalConfigs) totalConfigs - 1 else totalConfigs - mutationAggression
    val mutationCandidates = generateThresholdedParams(mutationCount)
    val mutationIndeces = generateMutationIndeces(1, totalConfigs, indexMutation, mutationCount)

    for (i <- mutationCandidates.indices) {

      val randomParent = scala.util.Random.shuffle(parents.toList).head
      val mutationIteration = mutationCandidates(i)
      val mutationIndexIteration = mutationIndeces(i)

      mutationPayload += SVMConfig(
        if (mutationIndexIteration.contains(0)) coinFlip(randomParent.fitIntercept,
          mutationIteration.fitIntercept, mutationMagnitude)
        else randomParent.fitIntercept,
        if (mutationIndexIteration.contains(1)) geneMixing(randomParent.maxIter,
          mutationIteration.maxIter, mutationMagnitude)
        else randomParent.maxIter,
        if (mutationIndexIteration.contains(2)) geneMixing(randomParent.regParam,
          mutationIteration.regParam, mutationMagnitude)
        else randomParent.regParam,
        if (mutationIndexIteration.contains(3)) coinFlip(randomParent.standardization,
          mutationIteration.standardization, mutationMagnitude)
        else randomParent.standardization,
        if (mutationIndexIteration.contains(4)) geneMixing(randomParent.tol,
          mutationIteration.tol, mutationMagnitude)
        else randomParent.tol
      )
    }
    mutationPayload.result.toArray
  }

  def generateIdealParents(results: Array[SVMModelsWithResults]): Array[SVMConfig] = {
    val bestParents = new ArrayBuffer[SVMConfig]
    results.take(_numberOfParentsToRetain).map(x => {
      bestParents += x.modelHyperParams
    })
    bestParents.result.toArray
  }

  def evolveParameters(startingSeed: Option[SVMConfig] = None): Array[SVMModelsWithResults] = {

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[SVMModelsWithResults]

    val totalConfigs = modelConfigLength[SVMConfig]

    val primordial = startingSeed match {
      case Some(`startingSeed`) =>
        val generativeArray = new ArrayBuffer[SVMConfig]
        generativeArray += startingSeed.asInstanceOf[SVMConfig]
        generativeArray ++= irradiateGeneration(
          Array(startingSeed.asInstanceOf[SVMConfig]),
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

  def evolveBest(startingSeed: Option[SVMConfig] = None): SVMModelsWithResults = {
    evolveParameters(startingSeed).head
  }

  def generateScoredDataFrame(results: Array[SVMModelsWithResults]): DataFrame = {

    import spark.sqlContext.implicits._

    val scoreBuffer = new ListBuffer[(Int, Double)]
    results.map(x => scoreBuffer += ((x.generation, x.score)))
    val scored = scoreBuffer.result
    spark.sparkContext.parallelize(scored)
      .toDF("generation", "score").orderBy(col("generation").asc, col("score").asc)
  }

  def evolveWithScoringDF(startingSeed: Option[SVMConfig] = None):
  (Array[SVMModelsWithResults], DataFrame) = {
    val evolutionResults = evolveParameters(startingSeed)
    (evolutionResults, generateScoredDataFrame(evolutionResults))
  }




}
