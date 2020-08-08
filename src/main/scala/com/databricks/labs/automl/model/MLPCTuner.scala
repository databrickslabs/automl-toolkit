package com.databricks.labs.automl.model

import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.model.tools.{
  GenerationOptimizer,
  HyperParameterFullSearch,
  ModelReporting
}
import com.databricks.labs.automl.params.{
  Defaults,
  MLPCConfig,
  MLPCModelsWithResults
}
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.col

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.collection.parallel.mutable.ParHashSet
import scala.concurrent.forkjoin.ForkJoinPool

class MLPCTuner(df: DataFrame,
                data: Array[TrainSplitReferences],
                isPipeline: Boolean = false)
    extends SparkSessionWrapper
    with Evolution
    with Defaults
    with AbstractTuner[MLPCConfig, MLPCModelsWithResults] {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _scoringMetric = _scoringDefaultClassifier
  private var _mlpcNumericBoundaries = _mlpcDefaultNumBoundaries
  private var _mlpcStringBoundaries = _mlpcDefaultStringBoundaries
  private var _featureInputSize: Int = 0
  private var _classDistinctCount: Int = 0
  private var _classificationMetrics = classificationMetrics

  private def calcFeatureInputSize: this.type = {
    _featureInputSize =
      df.select(_featureCol).head()(0).asInstanceOf[DenseVector].size
    this
  }

  private def calcClassDistinctCount: this.type = {
    _classDistinctCount = df.select(_labelCol).distinct().count().toInt
    this
  }

  def setScoringMetric(value: String): this.type = {
    require(
      classificationMetrics.contains(value),
      s"Classification scoring metric $value is not a valid member of ${invalidateSelection(value, classificationMetrics)}"
    )
    _scoringMetric = value
    this
  }

  def setMlpcNumericBoundaries(
    value: Map[String, (Double, Double)]
  ): this.type = {
    _mlpcNumericBoundaries = value
    this
  }

  def setMlpcStringBoundaries(value: Map[String, List[String]]): this.type = {
    _mlpcStringBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getMlpcNumericBoundaries: Map[String, (Double, Double)] =
    _mlpcNumericBoundaries

  def getMlpcStringBoundaries: Map[String, List[String]] = _mlpcStringBoundaries

  def getClassificationMetrics: List[String] = classificationMetrics

  def getFeatureInputSize: Int = _featureInputSize

  def getClassDistinctCount: Int = _classDistinctCount

  private def resetClassificationMetrics: List[String] =
    classificationMetricValidator(
      classificationAdjudicator(df),
      classificationMetrics
    )

  private def setClassificationMetrics(value: List[String]): this.type = {
    _classificationMetrics = value
    this
  }

  private def configureModel(
    modelConfig: MLPCConfig
  ): MultilayerPerceptronClassifier = {
    new MultilayerPerceptronClassifier()
      .setLabelCol(_labelCol)
      .setFeaturesCol(_featureCol)
      .setLayers(modelConfig.layers)
      .setMaxIter(modelConfig.maxIter)
      .setSolver(modelConfig.solver)
      .setStepSize(modelConfig.stepSize)
      .setTol(modelConfig.tolerance)
  }

  private def returnBestHyperParameters(
    collection: ArrayBuffer[MLPCModelsWithResults]
  ): (MLPCConfig, Double) = {

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
    results: ArrayBuffer[MLPCModelsWithResults]
  ): Array[MLPCModelsWithResults] = {
    _optimizationStrategy match {
      case "minimize" => results.result.toArray.sortWith(_.score < _.score)
      case _          => results.result.toArray.sortWith(_.score > _.score)
    }
  }

  private def sortAndReturnBestScore(
    results: ArrayBuffer[MLPCModelsWithResults]
  ): Double = {
    sortAndReturnAll(results).head.score
  }

  private def generateThresholdedParams(
    iterationCount: Int
  ): Array[MLPCConfig] = {

    val iterations = new ArrayBuffer[MLPCConfig]

    var i = 0
    do {
      val layers = generateLayerArray(
        "layers",
        "hiddenLayerSizeAdjust",
        _mlpcNumericBoundaries,
        _featureInputSize,
        _classDistinctCount + 1
      )
      val maxIter = generateRandomInteger("maxIter", _mlpcNumericBoundaries)
      val solver = generateRandomString("solver", _mlpcStringBoundaries)
      val stepSize = generateRandomDouble("stepSize", _mlpcNumericBoundaries)
      val tolerance = generateRandomDouble("tolerance", _mlpcNumericBoundaries)
      iterations += MLPCConfig(layers, maxIter, solver, stepSize, tolerance)
      i += 1
    } while (i < iterationCount)
    iterations.toArray
  }

  private def generateAndScoreMLPCModel(
    train: DataFrame,
    test: DataFrame,
    modelConfig: MLPCConfig,
    generation: Int = 1
  ): MLPCModelsWithResults = {

    val mlpcModel = configureModel(modelConfig)
    val builtModel = mlpcModel.fit(train)
    val predictedData = builtModel.transform(test)
    val optimizedPredictions = predictedData.persist(StorageLevel.DISK_ONLY)
//    optimizedPredictions.foreach(_ => ())

    val scoringMap = scala.collection.mutable.Map[String, Double]()

    for (i <- _classificationMetrics) {
      scoringMap(i) = classificationScoring(i, _labelCol, optimizedPredictions)
    }

    val mlpcModelsWithResults = MLPCModelsWithResults(
      modelConfig,
      builtModel,
      scoringMap(_scoringMetric),
      scoringMap.toMap,
      generation
    )

    optimizedPredictions.unpersist()
    mlpcModelsWithResults
  }

  private def runBattery(battery: Array[MLPCConfig],
                         generation: Int = 1): Array[MLPCModelsWithResults] = {

    val statusObj = new ModelReporting("mlpc", _classificationMetrics)

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    @volatile var results = new ArrayBuffer[MLPCModelsWithResults]
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

      val kFoldBuffer = data.map { z =>
        generateAndScoreMLPCModel(z.data.train, z.data.test, x)
      }

      val scores = kFoldBuffer.map(_.score)

      val scoringMap = scala.collection.mutable.Map[String, Double]()
      for (a <- _classificationMetrics) {
        val metricScores = new ListBuffer[Double]
        kFoldBuffer.map(x => metricScores += x.evalMetrics(a))
        scoringMap(a) = metricScores.sum / metricScores.length
      }

      val runAvg = MLPCModelsWithResults(
        x,
        kFoldBuffer.head.model,
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
    parents: Array[MLPCConfig],
    mutationCount: Int,
    mutationAggression: Int,
    mutationMagnitude: Double
  ): Array[MLPCConfig] = {

    val mutationPayload = new ArrayBuffer[MLPCConfig]
    val totalConfigs = modelConfigLength[MLPCConfig]
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

      mutationPayload += MLPCConfig(
        if (mutationIndexIteration.contains(0))
          geneMixing(
            randomParent.layers,
            mutationIteration.layers,
            mutationMagnitude
          )
        else randomParent.layers,
        if (mutationIndexIteration.contains(1))
          geneMixing(
            randomParent.maxIter,
            mutationIteration.maxIter,
            mutationMagnitude
          )
        else randomParent.maxIter,
        if (mutationIndexIteration.contains(2))
          geneMixing(randomParent.solver, mutationIteration.solver)
        else randomParent.solver,
        if (mutationIndexIteration.contains(3))
          geneMixing(
            randomParent.stepSize,
            mutationIteration.stepSize,
            mutationMagnitude
          )
        else randomParent.stepSize,
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

  private def continuousEvolution(): Array[MLPCModelsWithResults] = {

    setClassificationMetrics(resetClassificationMetrics)

    logger.log(Level.DEBUG, debugSettings)

    // Set the parameter guides for layers / label counts (only set once)
    calcFeatureInputSize
    calcClassDistinctCount

    val taskSupport = new ForkJoinTaskSupport(
      new ForkJoinPool(_continuousEvolutionParallelism)
    )

    var runResults = new ArrayBuffer[MLPCModelsWithResults]

    var scoreHistory = new ArrayBuffer[Double]

    // Set the beginning of the loop and instantiate a place holder for holdling the current best score
    var iter: Int = 1
    var bestScore: Double = 0.0
    var rollingImprovement: Boolean = true
    var incrementalImprovementCount: Int = 0

    val earlyStoppingImprovementThreshold: Int =
      _continuousEvolutionImprovementThreshold

    val totalConfigs = modelConfigLength[MLPCConfig]

    var runSet = _initialGenerationMode match {

      case "random" =>
        if (_modelSeedSet) {
          val genArray = new ArrayBuffer[MLPCConfig]
          val startingModelSeed = generateMLPCConfig(_modelSeed)
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
          .setModelFamily("MLPC")
          .setModelType("classifier")
          .setPermutationCount(_initialGenerationPermutationCount)
          .setIndexMixingMode(_initialGenerationIndexMixingMode)
          .setArraySeed(_initialGenerationArraySeed)
          .initialGenerationSeedMLPC(
            _mlpcNumericBoundaries,
            _mlpcStringBoundaries,
            _featureInputSize,
            _classDistinctCount
          )
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
    results: Array[MLPCModelsWithResults]
  ): Array[MLPCConfig] = {
    val bestParents = new ArrayBuffer[MLPCConfig]
    results
      .take(_numberOfParentsToRetain)
      .map(x => {
        bestParents += x.modelHyperParams
      })
    bestParents.result.toArray
  }

  def evolveParameters(): Array[MLPCModelsWithResults] = {

    setClassificationMetrics(resetClassificationMetrics)

    logger.log(Level.DEBUG, debugSettings)

    // Set the parameter guides for layers / label counts (only set once)
    this.calcFeatureInputSize
    this.calcClassDistinctCount

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[MLPCModelsWithResults]

    val totalConfigs = modelConfigLength[MLPCConfig]

    val primordial = _initialGenerationMode match {

      case "random" =>
        if (_modelSeedSet) {
          val generativeArray = new ArrayBuffer[MLPCConfig]
          val startingModelSeed = generateMLPCConfig(_modelSeed)
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
          .setModelFamily("MLPC")
          .setModelType("classifier")
          .setPermutationCount(_initialGenerationPermutationCount)
          .setIndexMixingMode(_initialGenerationIndexMixingMode)
          .setArraySeed(_initialGenerationArraySeed)
          .initialGenerationSeedMLPC(
            _mlpcNumericBoundaries,
            _mlpcStringBoundaries,
            this.getFeatureInputSize,
            this.getClassDistinctCount
          )
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
            .mlpcCandidates(
              "MLPC",
              _geneticMBORegressorType,
              fossilRecord,
              expandedCandidates,
              _optimizationStrategy,
              _numberOfMutationsPerGeneration,
              _featureInputSize,
              _classDistinctCount
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
          .mlpcCandidates(
            "MLPC",
            _geneticMBORegressorType,
            fossilRecord,
            expandedCandidates,
            _optimizationStrategy,
            _numberOfMutationsPerGeneration,
            _featureInputSize,
            _classDistinctCount
          )

        var evolve = runBattery(evolution, generation)
        generation += 1
        fossilRecord ++= evolve

      })

      sortAndReturnAll(fossilRecord)

    }
  }

  def evolveBest(): MLPCModelsWithResults = {
    evolveParameters().head
  }

  def generateScoredDataFrame(
    results: Array[MLPCModelsWithResults]
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

  def evolveWithScoringDF(): (Array[MLPCModelsWithResults], DataFrame) = {

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
    * @param paramsToTest Array of MLPC Configuration (hyper parameter settings) from the post-run model
    *                     inference
    * @return The results of the hyper parameter test, as well as the scored DataFrame report.
    */
  override def postRunModeledHyperParams(
    paramsToTest: Array[MLPCConfig]
  ): (Array[MLPCModelsWithResults], DataFrame) = {

    val finalRunResults =
      runBattery(paramsToTest, _numberOfMutationGenerations + 2)

    (finalRunResults, generateScoredDataFrame(finalRunResults))
  }

}
