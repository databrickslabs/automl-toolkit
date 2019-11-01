package com.databricks.labs.automl.model

import com.databricks.labs.automl.model.tools.{
  GenerationOptimizer,
  HyperParameterFullSearch,
  ModelReporting
}
import com.databricks.labs.automl.params
import com.databricks.labs.automl.params.{
  Defaults,
  TreesConfig,
  TreesModelsWithResults
}
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.col

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.collection.parallel.mutable.ParHashSet
import scala.concurrent.forkjoin.ForkJoinPool

class DecisionTreeTuner(df: DataFrame, modelSelection: String)
    extends SparkSessionWrapper
    with Evolution
    with Defaults {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _scoringMetric = modelSelection match {
    case "regressor"  => "rmse"
    case "classifier" => "f1"
    case _ =>
      throw new UnsupportedOperationException(
        s"Model $modelSelection is not supported."
      )
  }

  private var _treesNumericBoundaries = _treesDefaultNumBoundaries

  private var _treesStringBoundaries = _treesDefaultStringBoundaries

  private var _classificationMetrics = classificationMetrics

  def setScoringMetric(value: String): this.type = {
    modelSelection match {
      case "regressor" =>
        require(
          regressionMetrics.contains(value),
          s"Regressor scoring metric '$value' is not a valid member of ${invalidateSelection(value, regressionMetrics)}"
        )
      case "classifier" =>
        require(
          classificationMetrics.contains(value),
          s"Regressor scoring metric '$value' is not a valid member of ${invalidateSelection(value, classificationMetrics)}"
        )
      case _ =>
        throw new UnsupportedOperationException(
          s"Unsupported modelType $modelSelection"
        )
    }
    this._scoringMetric = value
    this
  }

  def setTreesNumericBoundaries(
    value: Map[String, (Double, Double)]
  ): this.type = {
    _treesNumericBoundaries = value
    this
  }

  def setTreesStringBoundaries(value: Map[String, List[String]]): this.type = {
    _treesStringBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getTreesNumericBoundaries: Map[String, (Double, Double)] =
    _treesNumericBoundaries

  def getTreesStringBoundaries: Map[String, List[String]] =
    _treesStringBoundaries

  def getClassificationMetrics: List[String] = classificationMetrics

  def getRegressionMetrics: List[String] = regressionMetrics

  private def resetClassificationMetrics: List[String] = modelSelection match {
    case "classifier" =>
      classificationMetricValidator(
        classificationAdjudicator(df),
        classificationMetrics
      )
    case _ => classificationMetrics
  }

  private def setClassificationMetrics(value: List[String]): this.type = {
    _classificationMetrics = value
    this
  }

  private def modelDecider[A, B](modelConfig: TreesConfig) = {

    val builtModel = modelSelection match {
      case "classifier" =>
        new DecisionTreeClassifier()
          .setLabelCol(_labelCol)
          .setFeaturesCol(_featureCol)
          .setMaxBins(modelConfig.maxBins)
          .setImpurity(modelConfig.impurity)
          .setMaxDepth(modelConfig.maxDepth)
          .setMinInfoGain(modelConfig.minInfoGain)
          .setMinInstancesPerNode(modelConfig.minInstancesPerNode)
      case "regressor" =>
        new DecisionTreeRegressor()
          .setLabelCol(_labelCol)
          .setFeaturesCol(_featureCol)
          .setMaxBins(modelConfig.maxBins)
          .setImpurity(modelConfig.impurity)
          .setMaxDepth(modelConfig.maxDepth)
          .setMinInfoGain(modelConfig.minInfoGain)
          .setMinInstancesPerNode(modelConfig.minInstancesPerNode)
      case _ =>
        throw new UnsupportedOperationException(
          s"Unsupported model type $modelSelection"
        )
    }
    builtModel
  }

  override def generateRandomString(
    param: String,
    boundaryMap: Map[String, List[String]]
  ): String = {

    val stringListing = param match {
      case "impurity" =>
        modelSelection match {
          case "regressor" => List("variance")
          case _           => boundaryMap(param)
        }
      case _ => boundaryMap(param)
    }
    _randomizer.shuffle(stringListing).head
  }

  private def returnBestHyperParameters(
    collection: ArrayBuffer[TreesModelsWithResults]
  ): (TreesConfig, Double) = {

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
    results: ArrayBuffer[TreesModelsWithResults]
  ): Array[TreesModelsWithResults] = {
    _optimizationStrategy match {
      case "minimize" => results.result.toArray.sortWith(_.score < _.score)
      case _          => results.result.toArray.sortWith(_.score > _.score)
    }
  }

  private def sortAndReturnBestScore(
    results: ArrayBuffer[TreesModelsWithResults]
  ): Double = {
    sortAndReturnAll(results).head.score
  }

  private def generateThresholdedParams(
    iterationCount: Int
  ): Array[TreesConfig] = {

    val iterations = new ArrayBuffer[TreesConfig]

    var i = 0
    do {
      val impurity = generateRandomString("impurity", _treesStringBoundaries)
      val maxBins = generateRandomInteger("maxBins", _treesNumericBoundaries)
      val maxDepth = generateRandomInteger("maxDepth", _treesNumericBoundaries)
      val minInfoGain =
        generateRandomDouble("minInfoGain", _treesNumericBoundaries)
      val minInstancesPerNode =
        generateRandomInteger("minInstancesPerNode", _treesNumericBoundaries)

      iterations += TreesConfig(
        impurity,
        maxBins,
        maxDepth,
        minInfoGain,
        minInstancesPerNode
      )
      i += 1
    } while (i < iterationCount)

    iterations.toArray
  }

  private def generateAndScoreTreesModel(
    train: DataFrame,
    test: DataFrame,
    modelConfig: TreesConfig,
    generation: Int = 1
  ): TreesModelsWithResults = {

    val treesModel = modelDecider(modelConfig)

    val builtModel = treesModel.fit(train)

    val predictedData = builtModel.transform(test)

    val scoringMap = scala.collection.mutable.Map[String, Double]()

    modelSelection match {
      case "classifier" =>
        for (i <- _classificationMetrics) {
          scoringMap(i) = classificationScoring(i, _labelCol, predictedData)
        }
      case "regressor" =>
        for (i <- regressionMetrics) {
          scoringMap(i) = regressionScoring(i, _labelCol, predictedData)
        }
    }

    TreesModelsWithResults(
      modelConfig,
      builtModel,
      scoringMap(_scoringMetric),
      scoringMap.toMap,
      generation
    )
  }

  private def runBattery(battery: Array[TreesConfig],
                         generation: Int = 1): Array[TreesModelsWithResults] = {

    val metrics = modelSelection match {
      case "classifier" => _classificationMetrics
      case _            => regressionMetrics
    }

    val statusObj = new ModelReporting("trees", metrics)

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    @volatile var results = new ArrayBuffer[TreesModelsWithResults]
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

      val kFoldBuffer = new ArrayBuffer[TreesModelsWithResults]

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) =
          genTestTrain(df, scala.util.Random.nextLong, uniqueLabels)
        kFoldBuffer += generateAndScoreTreesModel(train, test, x)
      }
      val scores = new ArrayBuffer[Double]
      kFoldBuffer.map(x => {
        scores += x.score
      })

      val scoringMap = scala.collection.mutable.Map[String, Double]()
      modelSelection match {
        case "classifier" =>
          for (a <- _classificationMetrics) {
            val metricScores = new ListBuffer[Double]
            kFoldBuffer.map(x => metricScores += x.evalMetrics(a))
            scoringMap(a) = metricScores.sum / metricScores.length
          }
        case "regressor" =>
          for (a <- regressionMetrics) {
            val metricScores = new ListBuffer[Double]
            kFoldBuffer.map(x => metricScores += x.evalMetrics(a))
            scoringMap(a) = metricScores.sum / metricScores.length
          }
        case _ =>
          throw new UnsupportedOperationException(
            s"$modelSelection is not a supported model type."
          )
      }

      val runAvg = params.TreesModelsWithResults(
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
    parents: Array[TreesConfig],
    mutationCount: Int,
    mutationAggression: Int,
    mutationMagnitude: Double
  ): Array[TreesConfig] = {

    val mutationPayload = new ArrayBuffer[TreesConfig]
    val totalConfigs = modelConfigLength[TreesConfig]
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

      mutationPayload += TreesConfig(
        if (mutationIndexIteration.contains(0))
          geneMixing(randomParent.impurity, mutationIteration.impurity)
        else randomParent.impurity,
        if (mutationIndexIteration.contains(1))
          geneMixing(
            randomParent.maxBins,
            mutationIteration.maxBins,
            mutationMagnitude
          )
        else randomParent.maxBins,
        if (mutationIndexIteration.contains(2))
          geneMixing(
            randomParent.maxDepth,
            mutationIteration.maxDepth,
            mutationMagnitude
          )
        else randomParent.maxDepth,
        if (mutationIndexIteration.contains(3))
          geneMixing(
            randomParent.minInfoGain,
            mutationIteration.minInfoGain,
            mutationMagnitude
          )
        else randomParent.minInfoGain,
        if (mutationIndexIteration.contains(4))
          geneMixing(
            randomParent.minInstancesPerNode,
            mutationIteration.minInstancesPerNode,
            mutationMagnitude
          )
        else randomParent.minInstancesPerNode
      )
    }
    mutationPayload.result.toArray
  }

  private def continuousEvolution(): Array[TreesModelsWithResults] = {

    setClassificationMetrics(resetClassificationMetrics)

    val taskSupport = new ForkJoinTaskSupport(
      new ForkJoinPool(_continuousEvolutionParallelism)
    )

    var runResults = new ArrayBuffer[TreesModelsWithResults]

    var scoreHistory = new ArrayBuffer[Double]

    // Set the beginning of the loop and instantiate a place holder for holdling the current best score
    var iter: Int = 1
    var bestScore: Double = 0.0
    var rollingImprovement: Boolean = true
    var incrementalImprovementCount: Int = 0
    val earlyStoppingImprovementThreshold: Int =
      _continuousEvolutionImprovementThreshold

    val totalConfigs = modelConfigLength[TreesConfig]
    // Generate the first pool of attempts to seed the hyperparameter space

    var runSet = _initialGenerationMode match {

      case "random" =>
        if (_modelSeedSet) {
          val genArray = new ArrayBuffer[TreesConfig]
          val startingModelSeed = generateTreesConfig(_modelSeed)
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
          .setModelFamily("Trees")
          .setModelType(modelSelection)
          .setPermutationCount(_initialGenerationPermutationCount)
          .setIndexMixingMode(_initialGenerationIndexMixingMode)
          .setArraySeed(_initialGenerationArraySeed)
          .initialGenerationSeedTrees(
            _treesNumericBoundaries,
            _treesStringBoundaries
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
    results: Array[TreesModelsWithResults]
  ): Array[TreesConfig] = {
    val bestParents = new ArrayBuffer[TreesConfig]
    results
      .take(_numberOfParentsToRetain)
      .map(x => {
        bestParents += x.modelHyperParams
      })
    bestParents.result.toArray
  }

  def evolveParameters(): Array[TreesModelsWithResults] = {

    setClassificationMetrics(resetClassificationMetrics)

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[TreesModelsWithResults]

    val totalConfigs = modelConfigLength[TreesConfig]

    val primordial = _initialGenerationMode match {

      case "random" =>
        if (_modelSeedSet) {
          val generativeArray = new ArrayBuffer[TreesConfig]
          val startingModelSeed = generateTreesConfig(_modelSeed)
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
          .setModelFamily("Trees")
          .setModelType(modelSelection)
          .setPermutationCount(_initialGenerationPermutationCount)
          .setIndexMixingMode(_initialGenerationIndexMixingMode)
          .setArraySeed(_initialGenerationArraySeed)
          .initialGenerationSeedTrees(
            _treesNumericBoundaries,
            _treesStringBoundaries
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
            .decisionTreesCandidates(
              "Trees",
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
          .decisionTreesCandidates(
            "Trees",
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

  def evolveBest(): TreesModelsWithResults = {
    evolveParameters().head
  }

  def generateScoredDataFrame(
    results: Array[TreesModelsWithResults]
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

  def evolveWithScoringDF(): (Array[TreesModelsWithResults], DataFrame) = {

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
    * @param paramsToTest Array of Decision Trees Configuration (hyper parameter settings) from the post-run model
    *                     inference
    * @return The results of the hyper parameter test, as well as the scored DataFrame report.
    */
  def postRunModeledHyperParams(
    paramsToTest: Array[TreesConfig]
  ): (Array[TreesModelsWithResults], DataFrame) = {

    val finalRunResults =
      runBattery(paramsToTest, _numberOfMutationGenerations + 2)

    (finalRunResults, generateScoredDataFrame(finalRunResults))
  }

}
