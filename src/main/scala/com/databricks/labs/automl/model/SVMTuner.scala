package com.databricks.labs.automl.model

import com.databricks.labs.automl.model.tools.{
  GenerationOptimizer,
  HyperParameterFullSearch,
  ModelReporting
}
import com.databricks.labs.automl.params.{
  Defaults,
  SVMConfig,
  SVMModelsWithResults
}
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.col

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.collection.parallel.mutable.ParHashSet
import scala.concurrent.forkjoin.ForkJoinPool

class SVMTuner(df: DataFrame)
    extends SparkSessionWrapper
    with Evolution
    with Defaults {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _scoringMetric = _scoringDefaultRegressor

  private var _svmNumericBoundaries = _svmDefaultNumBoundaries

  def setScoringMetric(value: String): this.type = {
    require(
      regressionMetrics.contains(value),
      s"Regressor scoring metric '$value' is not a valid member of ${invalidateSelection(value, regressionMetrics)}"
    )
    _scoringMetric = value
    this
  }

  def setSvmNumericBoundaries(
    value: Map[String, (Double, Double)]
  ): this.type = {
    _svmNumericBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getSvmNumericBoundaries: Map[String, (Double, Double)] =
    _svmNumericBoundaries

  def getRegressionMetrics: List[String] = regressionMetrics

  private def configureModel(modelConfig: SVMConfig): LinearSVC = {
    new LinearSVC()
      .setLabelCol(_labelCol)
      .setFeaturesCol(_featureCol)
      .setFitIntercept(modelConfig.fitIntercept)
      .setMaxIter(modelConfig.maxIter)
      .setRegParam(modelConfig.regParam)
      .setStandardization(modelConfig.standardization)
      .setTol(modelConfig.tolerance)
  }

  private def returnBestHyperParameters(
    collection: ArrayBuffer[SVMModelsWithResults]
  ): (SVMConfig, Double) = {

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
    results: ArrayBuffer[SVMModelsWithResults]
  ): Array[SVMModelsWithResults] = {
    _optimizationStrategy match {
      case "minimize" => results.result.toArray.sortWith(_.score < _.score)
      case _          => results.result.toArray.sortWith(_.score > _.score)
    }
  }

  private def sortAndReturnBestScore(
    results: ArrayBuffer[SVMModelsWithResults]
  ): Double = {
    sortAndReturnAll(results).head.score
  }

  private def generateThresholdedParams(
    iterationCount: Int
  ): Array[SVMConfig] = {
    val iterations = new ArrayBuffer[SVMConfig]

    var i = 0
    do {
      val fitIntercept = coinFlip()
      val maxIter = generateRandomInteger("maxIter", _svmNumericBoundaries)
      val regParam = generateRandomDouble("regParam", _svmNumericBoundaries)
      val standardization = coinFlip()
      val tolerance = generateRandomDouble("tolerance", _svmNumericBoundaries)
      iterations += SVMConfig(
        fitIntercept,
        maxIter,
        regParam,
        standardization,
        tolerance
      )
    } while (i < iterationCount)
    iterations.toArray
  }

  private def generateAndScoreSVM(train: DataFrame,
                                  test: DataFrame,
                                  modelConfig: SVMConfig,
                                  generation: Int = 1): SVMModelsWithResults = {

    val svmModel = configureModel(modelConfig)
    val builtModel = svmModel.fit(train)
    val predictedData = builtModel.transform(test)
    val scoringMap = scala.collection.mutable.Map[String, Double]()

    for (i <- regressionMetrics) {
      scoringMap(i) = regressionScoring(i, _labelCol, predictedData)
    }
    SVMModelsWithResults(
      modelConfig,
      builtModel,
      scoringMap(_scoringMetric),
      scoringMap.toMap,
      generation
    )
  }

  private def runBattery(battery: Array[SVMConfig],
                         generation: Int = 1): Array[SVMModelsWithResults] = {

    val statusObj = new ModelReporting("svm", regressionMetrics)

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    @volatile var results = new ArrayBuffer[SVMModelsWithResults]
    @volatile var modelCnt = 0
    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(_parallelism))
    val runs = battery.par
    runs.tasksupport = taskSupport

    val uniqueLabels: Array[Row] = df.select(_labelCol).distinct().collect()

    val currentStatus = statusObj.generateGenerationStartStatement(
      generation,
      calculateModelingFamilyRemainingTime(generation, modelCnt)
    )

    println(currentStatus)
    logger.log(Level.INFO, currentStatus)

    runs.foreach { x =>
      val runId = java.util.UUID.randomUUID()

      println(statusObj.generateRunStartStatement(runId, x))

      val kFoldTimeStamp = System.currentTimeMillis() / 1000

      val kFoldBuffer = new ArrayBuffer[SVMModelsWithResults]

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) =
          genTestTrain(df, scala.util.Random.nextLong, uniqueLabels)
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

      val runAvg = SVMModelsWithResults(
        x,
        kFoldBuffer.result.head.model,
        scores.sum / scores.length,
        scoringMap.toMap,
        generation
      )

      results += runAvg
      modelCnt += 1

      val runStatement = statusObj.generateRunScoreStatement(
        runId,
        scoringMap.result.toMap,
        _scoringMetric,
        x,
        calculateModelingFamilyRemainingTime(generation, modelCnt),
        kFoldTimeStamp
      )

      println(runStatement)

      logger.log(Level.INFO, runStatement)
    }

    sortAndReturnAll(results)

  }

  private def irradiateGeneration(
    parents: Array[SVMConfig],
    mutationCount: Int,
    mutationAggression: Int,
    mutationMagnitude: Double
  ): Array[SVMConfig] = {

    val mutationPayload = new ArrayBuffer[SVMConfig]
    val totalConfigs = modelConfigLength[SVMConfig]
    val indexMutation =
      if (mutationAggression >= totalConfigs) totalConfigs - 1
      else totalConfigs - mutationAggression
    val mutationCandidates = generateThresholdedParams(mutationCount)
    val mutationIndeces =
      generateMutationIndeces(1, totalConfigs, indexMutation, mutationCount)

    for (i <- mutationCandidates.indices) {

      val randomParent = scala.util.Random.shuffle(parents.toList).head
      val mutationIteration = mutationCandidates(i)
      val mutationIndexIteration = mutationIndeces(i)

      mutationPayload += SVMConfig(
        if (mutationIndexIteration.contains(0))
          coinFlip(
            randomParent.fitIntercept,
            mutationIteration.fitIntercept,
            mutationMagnitude
          )
        else randomParent.fitIntercept,
        if (mutationIndexIteration.contains(1))
          geneMixing(
            randomParent.maxIter,
            mutationIteration.maxIter,
            mutationMagnitude
          )
        else randomParent.maxIter,
        if (mutationIndexIteration.contains(2))
          geneMixing(
            randomParent.regParam,
            mutationIteration.regParam,
            mutationMagnitude
          )
        else randomParent.regParam,
        if (mutationIndexIteration.contains(3))
          coinFlip(
            randomParent.standardization,
            mutationIteration.standardization,
            mutationMagnitude
          )
        else randomParent.standardization,
        if (mutationIndexIteration.contains(4))
          geneMixing(
            randomParent.tolerance,
            mutationIteration.tolerance,
            mutationMagnitude
          )
        else randomParent.tolerance
      )
    }
    mutationPayload.result.toArray
  }

  private def continuousEvolution(): Array[SVMModelsWithResults] = {

    val taskSupport = new ForkJoinTaskSupport(
      new ForkJoinPool(_continuousEvolutionParallelism)
    )

    var runResults = new ArrayBuffer[SVMModelsWithResults]

    var scoreHistory = new ArrayBuffer[Double]

    // Set the beginning of the loop and instantiate a place holder for holdling the current best score
    var iter: Int = 1
    var bestScore: Double = 0.0
    var rollingImprovement: Boolean = true
    var incrementalImprovementCount: Int = 0
    val earlyStoppingImprovementThreshold: Int =
      _continuousEvolutionImprovementThreshold

    val totalConfigs = modelConfigLength[SVMConfig]

    var runSet = _initialGenerationMode match {

      case "random" =>
        if (_modelSeedSet) {
          val genArray = new ArrayBuffer[SVMConfig]
          val startingModelSeed = generateSVMConfig(_modelSeed)
          genArray += startingModelSeed
          genArray ++= irradiateGeneration(
            Array(startingModelSeed),
            _firstGenerationGenePool,
            totalConfigs - 1,
            _geneticMixing
          )
          ParHashSet(genArray.result.toArray: _*)
        } else {
          ParHashSet(generateThresholdedParams(_firstGenerationGenePool): _*)
        }
      case "permutations" =>
        val startingPool = new HyperParameterFullSearch()
          .setModelFamily("SVM")
          .setModelType("regressor")
          .setPermutationCount(_initialGenerationPermutationCount)
          .setIndexMixingMode(_initialGenerationIndexMixingMode)
          .setArraySeed(_initialGenerationArraySeed)
          .initialGenerationSeedSVM(_svmNumericBoundaries)
        ParHashSet(startingPool: _*)
    }

    // Apply ForkJoin ThreadPool parallelism
    runSet.tasksupport = taskSupport

    do {

      runSet.foreach(x => {

        try {
          // Pull the config out of the HashSet
          runSet -= x

          // Run the model config
          val run = runBattery(Array(x), iter)

          runResults += run.head
          scoreHistory += run.head.score

          val (bestConfig, currentBestScore) =
            returnBestHyperParameters(runResults)

          bestScore = currentBestScore

          // Add a mutated version of the current best model to the ParHashSet
          runSet += irradiateGeneration(
            Array(bestConfig),
            1,
            _continuousEvolutionMutationAggressiveness,
            _continuousEvolutionGeneticMixing
          ).head

          // Evaluate whether the scores are staying static over the last configured rolling window.
          val currentWindowValues = scoreHistory.slice(
            scoreHistory.length - _continuousEvolutionRollingImprovementCount,
            scoreHistory.length
          )

          // Check for static values
          val staticCheck = currentWindowValues.toSet.size

          // If there is more than one value, proceed with validation check on whether the model is improving over time.
          if (staticCheck > 1) {
            val (early, later) = currentWindowValues.splitAt(
              scala.math.round(currentWindowValues.size / 2)
            )
            if (later.sum / later.length < early.sum / early.length) {
              incrementalImprovementCount += 1
            } else {
              incrementalImprovementCount -= 1
            }
          } else {
            rollingImprovement = false
          }

          val statusReport = s"Current Best Score: $bestScore as of run: $iter with cumulative improvement count of: " +
            s"$incrementalImprovementCount"

          logger.log(Level.INFO, statusReport)
          println(statusReport)

          iter += 1

        } catch {
          case e: java.lang.NullPointerException =>
            val (bestConfig, currentBestScore) =
              returnBestHyperParameters(runResults)
            runSet += irradiateGeneration(
              Array(bestConfig),
              1,
              _continuousEvolutionMutationAggressiveness,
              _continuousEvolutionGeneticMixing
            ).head
            bestScore = currentBestScore
          case f: java.lang.ArrayIndexOutOfBoundsException =>
            val (bestConfig, currentBestScore) =
              returnBestHyperParameters(runResults)
            runSet += irradiateGeneration(
              Array(bestConfig),
              1,
              _continuousEvolutionMutationAggressiveness,
              _continuousEvolutionGeneticMixing
            ).head
            bestScore = currentBestScore
        }
      })
    } while (iter < _continuousEvolutionMaxIterations &&
      evaluateStoppingScore(bestScore, _continuousEvolutionStoppingScore)
      && rollingImprovement && incrementalImprovementCount > earlyStoppingImprovementThreshold)

    sortAndReturnAll(runResults)

  }

  def generateIdealParents(
    results: Array[SVMModelsWithResults]
  ): Array[SVMConfig] = {
    val bestParents = new ArrayBuffer[SVMConfig]
    results
      .take(_numberOfParentsToRetain)
      .map(x => {
        bestParents += x.modelHyperParams
      })
    bestParents.result.toArray
  }

  def evolveParameters(): Array[SVMModelsWithResults] = {

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[SVMModelsWithResults]

    val totalConfigs = modelConfigLength[SVMConfig]

    val primordial = _initialGenerationMode match {

      case "random" =>
        if (_modelSeedSet) {
          val generativeArray = new ArrayBuffer[SVMConfig]
          val startingModelSeed = generateSVMConfig(_modelSeed)
          generativeArray += startingModelSeed
          generativeArray ++= irradiateGeneration(
            Array(startingModelSeed),
            _firstGenerationGenePool,
            totalConfigs - 1,
            _geneticMixing
          )
          runBattery(generativeArray.result.toArray, generation)
        } else {
          runBattery(
            generateThresholdedParams(_firstGenerationGenePool),
            generation
          )
        }
      case "permutations" =>
        val startingPool = new HyperParameterFullSearch()
          .setModelFamily("RandomForest")
          .setModelType("regressor")
          .setPermutationCount(_initialGenerationPermutationCount)
          .setIndexMixingMode(_initialGenerationIndexMixingMode)
          .setArraySeed(_initialGenerationArraySeed)
          .initialGenerationSeedSVM(_svmNumericBoundaries)
        runBattery(startingPool, generation)
    }

    fossilRecord ++= primordial
    generation += 1

    var currentIteration = 1

    if (_earlyStoppingFlag) {

      var currentBestResult = sortAndReturnBestScore(fossilRecord)

      if (evaluateStoppingScore(currentBestResult, _earlyStoppingScore)) {
        while (currentIteration <= _numberOfMutationGenerations &&
               evaluateStoppingScore(currentBestResult, _earlyStoppingScore)) {

          val mutationAggressiveness: Int =
            generateAggressiveness(totalConfigs, currentIteration)

          // Get the sorted state
          val currentState = sortAndReturnAll(fossilRecord)

          val expandedCandidates = irradiateGeneration(
            generateIdealParents(currentState),
            _numberOfMutationsPerGeneration * _geneticMBOCandidateFactor,
            mutationAggressiveness,
            _geneticMixing
          )

          val evolution = GenerationOptimizer
            .svmCandidates(
              "SVM",
              _geneticMBORegressorType,
              fossilRecord,
              expandedCandidates,
              _optimizationStrategy,
              _numberOfMutationsPerGeneration
            )

          var evolve = runBattery(evolution, generation)
          generation += 1
          fossilRecord ++= evolve

          val postRunBestScore = sortAndReturnBestScore(fossilRecord)

          if (evaluateBestScore(postRunBestScore, currentBestResult))
            currentBestResult = postRunBestScore

          currentIteration += 1

        }

        sortAndReturnAll(fossilRecord)

      } else {
        sortAndReturnAll(fossilRecord)
      }
    } else {
      (1 to _numberOfMutationGenerations).map(i => {

        val mutationAggressiveness: Int =
          generateAggressiveness(totalConfigs, i)

        val currentState = sortAndReturnAll(fossilRecord)

        val expandedCandidates = irradiateGeneration(
          generateIdealParents(currentState),
          _numberOfMutationsPerGeneration * _geneticMBOCandidateFactor,
          mutationAggressiveness,
          _geneticMixing
        )

        val evolution = GenerationOptimizer
          .svmCandidates(
            "SVM",
            _geneticMBORegressorType,
            fossilRecord,
            expandedCandidates,
            _optimizationStrategy,
            _numberOfMutationsPerGeneration
          )

        var evolve = runBattery(evolution, generation)
        generation += 1
        fossilRecord ++= evolve

      })

      sortAndReturnAll(fossilRecord)

    }
  }

  def evolveBest(): SVMModelsWithResults = {
    require(df != null && df.count() > 0)
    evolveParameters().head
  }

  def generateScoredDataFrame(
    results: Array[SVMModelsWithResults]
  ): DataFrame = {

    import spark.sqlContext.implicits._

    val scoreBuffer = new ListBuffer[(Int, Double)]
    results.map(x => scoreBuffer += ((x.generation, x.score)))
    val scored = scoreBuffer.result
    spark.sparkContext
      .parallelize(scored)
      .toDF("generation", "score")
      .orderBy(col("generation").asc, col("score").asc)
  }

  def evolveWithScoringDF(): (Array[SVMModelsWithResults], DataFrame) = {

    val evolutionResults = _evolutionStrategy match {
      case "batch"      => evolveParameters()
      case "continuous" => continuousEvolution()
    }

    (evolutionResults, generateScoredDataFrame(evolutionResults))
  }

  /**
    * Helper Method for a post-run model optimization based on theoretical hyperparam multidimensional grid search space
    * After a genetic tuning run is complete, this allows for a model to be trained and run to predict a potential
    * best-condition of hyper parameter configurations.
    *
    * @param paramsToTest Array of SVM Configuration (hyper parameter settings) from the post-run model
    *                     inference
    * @return The results of the hyper parameter test, as well as the scored DataFrame report.
    */
  def postRunModeledHyperParams(
    paramsToTest: Array[SVMConfig]
  ): (Array[SVMModelsWithResults], DataFrame) = {

    val finalRunResults =
      runBattery(paramsToTest, _numberOfMutationGenerations + 2)

    (finalRunResults, generateScoredDataFrame(finalRunResults))
  }

}
