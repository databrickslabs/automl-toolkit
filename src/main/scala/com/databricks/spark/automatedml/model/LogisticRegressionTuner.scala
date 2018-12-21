package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.params.{Defaults, LogisticRegressionConfig, LogisticRegressionModelsWithResults}
import com.databricks.spark.automatedml.utils.SparkSessionWrapper
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

import org.apache.log4j.{Level, Logger}

class LogisticRegressionTuner(df: DataFrame) extends SparkSessionWrapper with Defaults
  with Evolution {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _scoringMetric = _scoringDefaultClassifier

  private var _logisticRegressionNumericBoundaries = _logisticRegressionDefaultNumBoundaries

  def setScoringMetric(value: String): this.type = {
    require(classificationMetrics.contains(value),
      s"Classification scoring metric $value is not a valid member of ${
        invalidateSelection(value, classificationMetrics)
      }")
    this._scoringMetric = value
    this
  }

  def setLogisticRegressionNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    this._logisticRegressionNumericBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getLogisticRegressionNumericBoundaries: Map[String, (Double, Double)] = _logisticRegressionNumericBoundaries

  private def configureModel(modelConfig: LogisticRegressionConfig): LogisticRegression = {
    new LogisticRegression()
      .setLabelCol(_labelCol)
      .setFeaturesCol(_featureCol)
      .setElasticNetParam(modelConfig.elasticNetParams)
      .setFamily("auto")
      .setFitIntercept(modelConfig.fitIntercept)
      .setMaxIter(modelConfig.maxIter)
      .setRegParam(modelConfig.regParam)
      .setStandardization(modelConfig.standardization)
      .setTol(modelConfig.tolerance)
  }

  private def generateThresholdedParams(iterationCount: Int): Array[LogisticRegressionConfig] = {

    val iterations = new ArrayBuffer[LogisticRegressionConfig]

    var i = 0
    do {
      val elasticNetParams = generateRandomDouble("elasticNetParams", _logisticRegressionNumericBoundaries)
      val fitIntercept = coinFlip()
      val maxIter = generateRandomInteger("maxIter", _logisticRegressionNumericBoundaries)
      val regParam = generateRandomDouble("regParam", _logisticRegressionNumericBoundaries)
      val standardization = coinFlip()
      val tolerance = generateRandomDouble("tolerance", _logisticRegressionNumericBoundaries)
      iterations += LogisticRegressionConfig(elasticNetParams, fitIntercept, maxIter, regParam, standardization,
        tolerance)
      i += 1
    } while (i < iterationCount)
    iterations.toArray
  }

  private def generateAndScoreLogisticRegression(train: DataFrame, test: DataFrame,
                                                 modelConfig: LogisticRegressionConfig,
                                                 generation: Int = 1): LogisticRegressionModelsWithResults = {
    val regressionModel = configureModel(modelConfig)

    val builtModel = regressionModel.fit(train)

    val predictedData = builtModel.transform(test)

    val scoringMap = scala.collection.mutable.Map[String, Double]()

    for (i <- classificationMetrics) {
      val scoreEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol(_labelCol)
        .setPredictionCol("prediction")
        .setMetricName(i)
      scoringMap(i) = scoreEvaluator.evaluate(predictedData)
    }
    LogisticRegressionModelsWithResults(modelConfig, builtModel, scoringMap(_scoringMetric), scoringMap.toMap,
      generation)
  }


  private def runBattery(battery: Array[LogisticRegressionConfig],
                 generation: Int = 1): Array[LogisticRegressionModelsWithResults] = {

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    @volatile var results = new ArrayBuffer[LogisticRegressionModelsWithResults]
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

      val kFoldBuffer = new ArrayBuffer[LogisticRegressionModelsWithResults]

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) = genTestTrain(df, scala.util.Random.nextLong)
        kFoldBuffer += generateAndScoreLogisticRegression(train, test, x)
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
      val runAvg = LogisticRegressionModelsWithResults(x, kFoldBuffer.result.head.model, scores.sum / scores.length,
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

  private def irradiateGeneration(parents: Array[LogisticRegressionConfig], mutationCount: Int,
                          mutationAggression: Int, mutationMagnitude: Double): Array[LogisticRegressionConfig] = {

    val mutationPayload = new ArrayBuffer[LogisticRegressionConfig]
    val totalConfigs = modelConfigLength[LogisticRegressionConfig]
    val indexMutation = if (mutationAggression >= totalConfigs) totalConfigs - 1 else totalConfigs - mutationAggression
    val mutationCandidates = generateThresholdedParams(mutationCount)
    val mutationIndeces = generateMutationIndeces(1, totalConfigs, indexMutation, mutationCount)

    for (i <- mutationCandidates.indices) {

      val randomParent = scala.util.Random.shuffle(parents.toList).head
      val mutationIteration = mutationCandidates(i)
      val mutationIndexIteration = mutationIndeces(i)

      mutationPayload += LogisticRegressionConfig(
        if (mutationIndexIteration.contains(0)) geneMixing(randomParent.elasticNetParams,
          mutationIteration.elasticNetParams, mutationMagnitude)
        else randomParent.elasticNetParams,
        if (mutationIndexIteration.contains(1)) coinFlip(randomParent.fitIntercept,
          mutationIteration.fitIntercept, mutationMagnitude)
        else randomParent.fitIntercept,
        if (mutationIndexIteration.contains(2)) geneMixing(randomParent.maxIter,
          mutationIteration.maxIter, mutationMagnitude)
        else randomParent.maxIter,
        if (mutationIndexIteration.contains(3)) geneMixing(randomParent.regParam,
          mutationIteration.regParam, mutationMagnitude)
        else randomParent.regParam,
        if (mutationIndexIteration.contains(4)) coinFlip(randomParent.standardization,
          mutationIteration.standardization, mutationMagnitude)
        else randomParent.standardization,
        if (mutationIndexIteration.contains(5)) geneMixing(randomParent.tolerance,
          mutationIteration.tolerance, mutationMagnitude)
        else randomParent.tolerance
      )
    }
    mutationPayload.result.toArray
  }

  def generateIdealParents(results: Array[LogisticRegressionModelsWithResults]): Array[LogisticRegressionConfig] = {
    val bestParents = new ArrayBuffer[LogisticRegressionConfig]
    results.take(_numberOfParentsToRetain).map(x => {
      bestParents += x.modelHyperParams
    })
    bestParents.result.toArray
  }

  def evolveParameters(startingSeed: Option[LogisticRegressionConfig] = None): Array[LogisticRegressionModelsWithResults] = {

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[LogisticRegressionModelsWithResults]

    val totalConfigs = modelConfigLength[LogisticRegressionConfig]

    val primordial = startingSeed match {
      case Some(`startingSeed`) =>
        val generativeArray = new ArrayBuffer[LogisticRegressionConfig]
        generativeArray += startingSeed.asInstanceOf[LogisticRegressionConfig]
        generativeArray ++= irradiateGeneration(
          Array(startingSeed.asInstanceOf[LogisticRegressionConfig]),
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

  def evolveBest(startingSeed: Option[LogisticRegressionConfig] = None): LogisticRegressionModelsWithResults = {
    evolveParameters(startingSeed).head
  }

  def generateScoredDataFrame(results: Array[LogisticRegressionModelsWithResults]): DataFrame = {

    import spark.sqlContext.implicits._

    val scoreBuffer = new ListBuffer[(Int, Double)]
    results.map(x => scoreBuffer += ((x.generation, x.score)))
    val scored = scoreBuffer.result
    spark.sparkContext.parallelize(scored)
      .toDF("generation", "score").orderBy(col("generation").asc, col("score").asc)
  }

  def evolveWithScoringDF(startingSeed: Option[LogisticRegressionConfig] = None):
  (Array[LogisticRegressionModelsWithResults], DataFrame) = {
    val evolutionResults = evolveParameters(startingSeed)
    (evolutionResults, generateScoredDataFrame(evolutionResults))
  }

}
