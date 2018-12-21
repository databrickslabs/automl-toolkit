package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.params.{Defaults, LinearRegressionConfig, LinearRegressionModelsWithResults}
import com.databricks.spark.automatedml.utils.SparkSessionWrapper
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

import org.apache.log4j.{Level, Logger}

class LinearRegressionTuner(df: DataFrame) extends SparkSessionWrapper with Defaults
  with Evolution {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _scoringMetric = _scoringDefaultRegressor
  private var _linearRegressionNumericBoundaries = _linearRegressionDefaultNumBoundaries
  private var _linearRegressionStringBoundaries = _linearRegressionDefaultStringBoundaries

  def setScoringMetric(value: String): this.type = {
    require(regressionMetrics.contains(value),
      s"Regressor scoring metric '$value' is not a valid member of ${
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
      .setElasticNetParam(modelConfig.elasticNetParams)
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
                                               modelConfig: LinearRegressionConfig,
                                               generation: Int = 1): LinearRegressionModelsWithResults = {

    val regressionModel = configureModel(modelConfig)

    val builtModel = regressionModel.fit(train)

    val predictedData = builtModel.transform(test)

    val scoringMap = scala.collection.mutable.Map[String, Double]()

    for (i <- regressionMetrics) {
      val scoreEvaluator = new RegressionEvaluator()
        .setLabelCol(_labelCol)
        .setPredictionCol("prediction")
        .setMetricName(i)
      scoringMap(i) = scoreEvaluator.evaluate(predictedData)
    }
    LinearRegressionModelsWithResults(modelConfig, builtModel, scoringMap(_scoringMetric),
      scoringMap.toMap, generation)
  }

  private def runBattery(battery: Array[LinearRegressionConfig], generation: Int = 1): Array[LinearRegressionModelsWithResults] = {

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    @volatile var results = new ArrayBuffer[LinearRegressionModelsWithResults]
    @volatile var modelCnt = 0
    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(_parallelism))
    val runs = battery.par
    runs.tasksupport = taskSupport

    val currentStatus = f"Starting Generation $generation \n\t\t Completion Status: ${
      calculateModelingFamilyRemainingTime(generation, modelCnt)}%2.4f%%"

    println(currentStatus)
    logger.log(Level.INFO, currentStatus)

    runs.foreach { x =>
      val runId = java.util.UUID.randomUUID()
      println(s"Starting run $runId with Params: ${x.toString}")

      val kFoldBuffer = new ArrayBuffer[LinearRegressionModelsWithResults]

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) = genTestTrain(df, scala.util.Random.nextLong)
        kFoldBuffer += generateAndScoreLinearRegression(train, test, x)
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
      val runAvg = LinearRegressionModelsWithResults(x, kFoldBuffer.result.head.model, scores.sum / scores.length,
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

  private def irradiateGeneration(parents: Array[LinearRegressionConfig], mutationCount: Int,
                          mutationAggression: Int, mutationMagnitude: Double): Array[LinearRegressionConfig] = {

    val mutationPayload = new ArrayBuffer[LinearRegressionConfig]
    val totalConfigs = modelConfigLength[LinearRegressionConfig]
    val indexMutation = if (mutationAggression >= totalConfigs) totalConfigs - 1 else totalConfigs - mutationAggression
    val mutationCandidates = generateThresholdedParams(mutationCount)
    val mutationIndeces = generateMutationIndeces(1, totalConfigs, indexMutation, mutationCount)

    for (i <- mutationCandidates.indices) {

      val randomParent = scala.util.Random.shuffle(parents.toList).head
      val mutationIteration = mutationCandidates(i)
      val mutationIndexIteration = mutationIndeces(i)

      mutationPayload += LinearRegressionConfig(
        if (mutationIndexIteration.contains(0)) geneMixing(randomParent.elasticNetParams,
          mutationIteration.elasticNetParams, mutationMagnitude)
        else randomParent.elasticNetParams,
        if (mutationIndexIteration.contains(1)) coinFlip(randomParent.fitIntercept,
          mutationIteration.fitIntercept, mutationMagnitude)
        else randomParent.fitIntercept,
        if (mutationIndexIteration.contains(2)) geneMixing(randomParent.loss,
          mutationIteration.loss)
        else randomParent.loss,
        if (mutationIndexIteration.contains(3)) geneMixing(randomParent.maxIter,
          mutationIteration.maxIter, mutationMagnitude)
        else randomParent.maxIter,
        if (mutationIndexIteration.contains(4)) geneMixing(randomParent.regParam,
          mutationIteration.regParam, mutationMagnitude)
        else randomParent.regParam,
        if (mutationIndexIteration.contains(5)) coinFlip(randomParent.standardization,
          mutationIteration.standardization, mutationMagnitude)
        else randomParent.standardization,
        if (mutationIndexIteration.contains(6)) geneMixing(randomParent.tolerance,
          mutationIteration.tolerance, mutationMagnitude)
        else randomParent.tolerance
      )
    }
    mutationPayload.result.toArray
  }

  def generateIdealParents(results: Array[LinearRegressionModelsWithResults]): Array[LinearRegressionConfig] = {
    val bestParents = new ArrayBuffer[LinearRegressionConfig]
    results.take(_numberOfParentsToRetain).map(x => {
      bestParents += x.modelHyperParams
    })
    bestParents.result.toArray
  }

  def evolveParameters(startingSeed: Option[LinearRegressionConfig] = None): Array[LinearRegressionModelsWithResults] = {

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[LinearRegressionModelsWithResults]

    val totalConfigs = modelConfigLength[LinearRegressionConfig]

    val primordial = startingSeed match {
      case Some(`startingSeed`) =>
        val generativeArray = new ArrayBuffer[LinearRegressionConfig]
        generativeArray += startingSeed.asInstanceOf[LinearRegressionConfig]
        generativeArray ++= irradiateGeneration(
          Array(startingSeed.asInstanceOf[LinearRegressionConfig]),
          _firstGenerationGenePool, totalConfigs - 1, _geneticMixing)
        runBattery(generativeArray.result.toArray, generation)
      case _ => runBattery(generateThresholdedParams(_firstGenerationGenePool), generation)
    }

    fossilRecord ++= primordial
    generation += 1

    var currentIteration = 1

    if(_earlyStoppingFlag) {

      _optimizationStrategy match {
        case "minimize" =>

          var currentBestResult: Double = fossilRecord.result.toArray.sortWith(_.score < _.score).head.score

          if (currentBestResult > _earlyStoppingScore) {
            while (currentIteration <= _numberOfMutationGenerations && currentBestResult > _earlyStoppingScore) {

              val mutationAggressiveness = _generationalMutationStrategy match {
                case "linear" => if (totalConfigs - (currentIteration + 1) < 1) 1 else totalConfigs - (currentIteration + 1)
                case _ => _fixedMutationValue
              }

              // Get the sorted state
              val currentState = fossilRecord.result.toArray.sortWith(_.score < _.score)

              val evolution = irradiateGeneration(generateIdealParents(currentState), _numberOfMutationsPerGeneration,
                mutationAggressiveness, _geneticMixing)

              var evolve = runBattery(evolution, generation)
              generation += 1
              fossilRecord ++= evolve

              val postRunBestScore = fossilRecord.result.toArray.sortWith(_.score < _.score).head.score

              if (postRunBestScore < currentBestResult) currentBestResult = postRunBestScore

              currentIteration += 1

            }

            fossilRecord.result.toArray.sortWith(_.score < _.score)
          } else {
            fossilRecord.result.toArray.sortWith(_.score < _.score)
          }
        case _ =>

          var currentBestResult: Double = fossilRecord.result.toArray.sortWith(_.score > _.score).head.score

          if (currentBestResult < _earlyStoppingScore) {
            while (currentIteration <= _numberOfMutationGenerations && currentBestResult < _earlyStoppingScore) {

              val mutationAggressiveness = _generationalMutationStrategy match {
                case "linear" => if (totalConfigs - (currentIteration + 1) < 1) 1 else totalConfigs - (currentIteration + 1)
                case _ => _fixedMutationValue
              }

              // Get the sorted state
              val currentState = fossilRecord.result.toArray.sortWith(_.score > _.score)

              val evolution = irradiateGeneration(generateIdealParents(currentState), _numberOfMutationsPerGeneration,
                mutationAggressiveness, _geneticMixing)

              var evolve = runBattery(evolution, generation)
              generation += 1
              fossilRecord ++= evolve

              val postRunBestScore = fossilRecord.result.toArray.sortWith(_.score > _.score).head.score

              if (postRunBestScore > currentBestResult) currentBestResult = postRunBestScore

              currentIteration += 1
              generation += 1

            }
            fossilRecord.result.toArray.sortWith(_.score > _.score)
          } else {
            fossilRecord.result.toArray.sortWith(_.score > _.score)
          }
      }
    } else {

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
  }

  def evolveBest(startingSeed: Option[LinearRegressionConfig] = None): LinearRegressionModelsWithResults = {
    evolveParameters(startingSeed).head
  }

  def generateScoredDataFrame(results: Array[LinearRegressionModelsWithResults]): DataFrame = {

    import spark.sqlContext.implicits._

    val scoreBuffer = new ListBuffer[(Int, Double)]
    results.map(x => scoreBuffer += ((x.generation, x.score)))
    val scored = scoreBuffer.result
    spark.sparkContext.parallelize(scored)
      .toDF("generation", "score").orderBy(col("generation").asc, col("score").asc)
  }

  def evolveWithScoringDF(startingSeed: Option[LinearRegressionConfig] = None):
  (Array[LinearRegressionModelsWithResults], DataFrame) = {
    val evolutionResults = evolveParameters(startingSeed)
    (evolutionResults, generateScoredDataFrame(evolutionResults))
  }


}
