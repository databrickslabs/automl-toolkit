package com.databricks.labs.automl.model

import com.databricks.labs.automl.model.tools.{
  GenerationOptimizer,
  HyperParameterFullSearch,
  ModelReporting
}
import com.databricks.labs.automl.params.{
  Defaults,
  XGBoostConfig,
  XGBoostModelsWithResults
}
import com.databricks.labs.automl.utils.SparkSessionWrapper
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostRegressor}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.collection.parallel.mutable.ParHashSet
import scala.concurrent.forkjoin.ForkJoinPool

class XGBoostTuner(df: DataFrame, modelSelection: String)
    extends SparkSessionWrapper
    with Evolution
    with Defaults
    with Serializable {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _scoringMetric = modelSelection match {
    case "regressor"  => "rmse"
    case "classifier" => "f1"
    case _ =>
      throw new UnsupportedOperationException(
        s"Model $modelSelection is not supported."
      )
  }

  private var _classificationMetrics = classificationMetrics

  private var _xgboostNumericBoundaries = _xgboostDefaultNumBoundaries

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

  def setXGBoostNumericBoundaries(
    value: Map[String, (Double, Double)]
  ): this.type = {
    _xgboostNumericBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getXGBoostNumericBoundaries: Map[String, (Double, Double)] =
    _xgboostNumericBoundaries

  def getClassificationMetrics: List[String] = _classificationMetrics

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

  final lazy val uniqueLabels: Int = modelSelection match {
    case "regressor" => 0
    case "classifier" =>
      df.select(col(_labelCol)).distinct.count.toInt
  }

  private def modelDecider[A, B](modelConfig: XGBoostConfig) = {

    val xgObjective: String = modelSelection match {
      case "regressor" => "None"
      case _ =>
        uniqueLabels match {
          case x if x <= 2 => "reg:squarederror"
          case _           => "multi:softmax"
        }
    }

    val builtModel = modelSelection match {
      case "classifier" =>
        val xgClass = new XGBoostClassifier()
          .setLabelCol(_labelCol)
          .setFeaturesCol(_featureCol)
          .setAlpha(modelConfig.alpha)
          .setEta(modelConfig.eta)
          .setGamma(modelConfig.gamma)
          .setLambda(modelConfig.lambda)
          .setMaxDepth(modelConfig.maxDepth)
          .setMaxBins(modelConfig.maxBins)
          .setSubsample(modelConfig.subSample)
          .setMinChildWeight(modelConfig.minChildWeight)
          .setNumRound(modelConfig.numRound)
          .setTrainTestRatio(modelConfig.trainTestRatio)
          .setMissing(0.0f)
        if (uniqueLabels > 2) {
          xgClass
            .setNumClass(uniqueLabels)
            .setObjective(xgObjective)
        }
        xgClass
      case "regressor" =>
        new XGBoostRegressor()
          .setLabelCol(_labelCol)
          .setFeaturesCol(_featureCol)
          .setAlpha(modelConfig.alpha)
          .setEta(modelConfig.eta)
          .setGamma(modelConfig.gamma)
          .setLambda(modelConfig.lambda)
          .setMaxDepth(modelConfig.maxDepth)
          .setMaxBins(modelConfig.maxBins)
          .setSubsample(modelConfig.subSample)
          .setMinChildWeight(modelConfig.minChildWeight)
          .setNumRound(modelConfig.numRound)
          .setTrainTestRatio(modelConfig.trainTestRatio)
          .setMissing(0.0f)
      case _ =>
        throw new UnsupportedOperationException(
          s"Unsupported modelType $modelSelection"
        )
    }
    builtModel
  }

  private def returnBestHyperParameters(
    collection: ArrayBuffer[XGBoostModelsWithResults]
  ): (XGBoostConfig, Double) = {

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
    results: ArrayBuffer[XGBoostModelsWithResults]
  ): Array[XGBoostModelsWithResults] = {
    _optimizationStrategy match {
      case "minimize" => results.result.toArray.sortWith(_.score < _.score)
      case _          => results.result.toArray.sortWith(_.score > _.score)
    }
  }

  private def sortAndReturnBestScore(
    results: ArrayBuffer[XGBoostModelsWithResults]
  ): Double = {
    sortAndReturnAll(results).head.score
  }

  /**
    * Method for extracting the predicted class for multi-class classification problems directly from the probabilities
    * linalg.Vector field.  This is due to a bug in XGBoost4j-spark and should be future-proof.
    * @param data The transformed data frame with the incorrect prediction values
    * @return Fixed prediction column that acquires the predicted class label from the probability Vector
    * @author Ben Wilson
    * @since 0.5.1
    */
  private def multiClassPredictionExtract(data: DataFrame): DataFrame = {

    // udf must be defined as a function in order to be serialized as an Object.  Defining as a method
    // prevents the Future from serializing properly.
    val extractUDF = udf(
      (v: org.apache.spark.ml.linalg.Vector) => v.toArray.last
    )
    // Replace the prediction column with the correct data.
    data.withColumn("prediction", extractUDF(col("probability")))
  }

  private def generateThresholdedParams(
    iterationCount: Int
  ): Array[XGBoostConfig] = {

    val iterations = new ArrayBuffer[XGBoostConfig]

    var i = 0
    do {
      val alpha = generateRandomDouble("alpha", _xgboostNumericBoundaries)
      val eta = generateRandomDouble("eta", _xgboostNumericBoundaries)
      val gamma = generateRandomDouble("gamma", _xgboostNumericBoundaries)
      val lambda = generateRandomDouble("lambda", _xgboostNumericBoundaries)
      val maxDepth =
        generateRandomInteger("maxDepth", _xgboostNumericBoundaries)
      val subSample =
        generateRandomDouble("subSample", _xgboostNumericBoundaries)
      val minChildWeight =
        generateRandomDouble("minChildWeight", _xgboostNumericBoundaries)
      val numRound =
        generateRandomInteger("numRound", _xgboostNumericBoundaries)
      val maxBins = generateRandomInteger("maxBins", _xgboostNumericBoundaries)
      val trainTestRatio =
        generateRandomDouble("trainTestRatio", _xgboostNumericBoundaries)
      iterations += XGBoostConfig(
        alpha,
        eta,
        gamma,
        lambda,
        maxDepth,
        subSample,
        minChildWeight,
        numRound,
        maxBins,
        trainTestRatio
      )
      i += 1
    } while (i < iterationCount)

    iterations.toArray
  }

  private def generateAndScoreXGBoostModel(
    train: DataFrame,
    test: DataFrame,
    modelConfig: XGBoostConfig,
    generation: Int = 1
  ): XGBoostModelsWithResults = {

    val xgboostModel = modelDecider(modelConfig)

    val builtModel = xgboostModel.fit(train)

    val predictedData = builtModel.transform(test)

    // Due to a bug in XGBoost's transformer for accessing the probability Vector to provide a prediction
    // This method needs to be called if the unique count for the label class is non-binary for a classifier.

    val fixedPredictionData = modelSelection match {
      case "regressor" => predictedData
      case _ =>
        uniqueLabels match {
          case x if x <= 2 => predictedData
          case _           => multiClassPredictionExtract(predictedData)
        }
    }

    val scoringMap = scala.collection.mutable.Map[String, Double]()

    modelSelection match {
      case "classifier" =>
        for (i <- _classificationMetrics) {
          scoringMap(i) =
            classificationScoring(i, _labelCol, fixedPredictionData)
        }
      case "regressor" =>
        for (i <- regressionMetrics) {
          scoringMap(i) = regressionScoring(i, _labelCol, fixedPredictionData)
        }
    }

    XGBoostModelsWithResults(
      modelConfig,
      builtModel,
      scoringMap(_scoringMetric),
      scoringMap.toMap,
      generation
    )
  }

  private def runBattery(
    battery: Array[XGBoostConfig],
    generation: Int = 1
  ): Array[XGBoostModelsWithResults] = {

    val metrics = modelSelection match {
      case "classifier" => _classificationMetrics
      case _            => regressionMetrics
    }

    val statusObj = new ModelReporting("xgboost", metrics)

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    @volatile var results = new ArrayBuffer[XGBoostModelsWithResults]
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

      val kFoldBuffer = new ArrayBuffer[XGBoostModelsWithResults]

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) =
          genTestTrain(df, scala.util.Random.nextLong, uniqueLabels)
        kFoldBuffer += generateAndScoreXGBoostModel(train, test, x)
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

      val runAvg = XGBoostModelsWithResults(
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
    parents: Array[XGBoostConfig],
    mutationCount: Int,
    mutationAggression: Int,
    mutationMagnitude: Double
  ): Array[XGBoostConfig] = {

    val mutationPayload = new ArrayBuffer[XGBoostConfig]
    val totalConfigs = modelConfigLength[XGBoostConfig]
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

      mutationPayload += XGBoostConfig(
        if (mutationIndexIteration.contains(0))
          geneMixing(
            randomParent.alpha,
            mutationIteration.alpha,
            mutationMagnitude
          )
        else randomParent.alpha,
        if (mutationIndexIteration.contains(1))
          geneMixing(randomParent.eta, mutationIteration.eta, mutationMagnitude)
        else randomParent.eta,
        if (mutationIndexIteration.contains(2))
          geneMixing(
            randomParent.gamma,
            mutationIteration.gamma,
            mutationMagnitude
          )
        else randomParent.gamma,
        if (mutationIndexIteration.contains(3))
          geneMixing(
            randomParent.lambda,
            mutationIteration.lambda,
            mutationMagnitude
          )
        else randomParent.lambda,
        if (mutationIndexIteration.contains(4))
          geneMixing(
            randomParent.maxDepth,
            mutationIteration.maxDepth,
            mutationMagnitude
          )
        else randomParent.maxDepth,
        if (mutationIndexIteration.contains(5))
          geneMixing(
            randomParent.subSample,
            mutationIteration.subSample,
            mutationMagnitude
          )
        else randomParent.subSample,
        if (mutationIndexIteration.contains(6))
          geneMixing(
            randomParent.minChildWeight,
            mutationIteration.minChildWeight,
            mutationMagnitude
          )
        else randomParent.minChildWeight,
        if (mutationIndexIteration.contains(7))
          geneMixing(
            randomParent.numRound,
            mutationIteration.numRound,
            mutationMagnitude
          )
        else randomParent.numRound,
        if (mutationIndexIteration.contains(8))
          geneMixing(
            randomParent.maxBins,
            mutationIteration.maxBins,
            mutationMagnitude
          )
        else randomParent.maxBins,
        if (mutationIndexIteration.contains(9))
          geneMixing(
            randomParent.trainTestRatio,
            mutationIteration.trainTestRatio,
            mutationMagnitude
          )
        else randomParent.trainTestRatio
      )
    }
    mutationPayload.result.toArray
  }

  private def continuousEvolution(): Array[XGBoostModelsWithResults] = {

    setClassificationMetrics(resetClassificationMetrics)

    val taskSupport = new ForkJoinTaskSupport(
      new ForkJoinPool(_continuousEvolutionParallelism)
    )

    var runResults = new ArrayBuffer[XGBoostModelsWithResults]

    var scoreHistory = new ArrayBuffer[Double]

    // Set the beginning of the loop and instantiate a place holder for holdling the current best score
    var iter: Int = 1
    var bestScore: Double = 0.0
    var rollingImprovement: Boolean = true
    var incrementalImprovementCount: Int = 0
    val earlyStoppingImprovementThreshold: Int =
      _continuousEvolutionImprovementThreshold

    val totalConfigs = modelConfigLength[XGBoostConfig]

    var runSet = _initialGenerationMode match {

      case "random" =>
        if (_modelSeedSet) {
          val genArray = new ArrayBuffer[XGBoostConfig]
          val startingModelSeed = generateXGBoostConfig(_modelSeed)
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
          .setModelFamily("XGBoost")
          .setModelType(modelSelection)
          .setPermutationCount(_initialGenerationPermutationCount)
          .setIndexMixingMode(_initialGenerationIndexMixingMode)
          .setArraySeed(_initialGenerationArraySeed)
          .initialGenerationSeedXGBoost(_xgboostNumericBoundaries)
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
    results: Array[XGBoostModelsWithResults]
  ): Array[XGBoostConfig] = {
    val bestParents = new ArrayBuffer[XGBoostConfig]
    results
      .take(_numberOfParentsToRetain)
      .map(x => {
        bestParents += x.modelHyperParams
      })
    bestParents.result.toArray
  }

  def evolveParameters(): Array[XGBoostModelsWithResults] = {

    setClassificationMetrics(resetClassificationMetrics)

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[XGBoostModelsWithResults]

    val totalConfigs = modelConfigLength[XGBoostConfig]

    val primordial = _initialGenerationMode match {

      case "random" =>
        if (_modelSeedSet) {
          val generativeArray = new ArrayBuffer[XGBoostConfig]
          val startingModelSeed = generateXGBoostConfig(_modelSeed)
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
          .setModelFamily("XGBoost")
          .setModelType(modelSelection)
          .setPermutationCount(_initialGenerationPermutationCount)
          .setIndexMixingMode(_initialGenerationIndexMixingMode)
          .setArraySeed(_initialGenerationArraySeed)
          .initialGenerationSeedXGBoost(_xgboostNumericBoundaries)
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
            .xgBoostCandidates(
              "XGBoost",
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
          .xgBoostCandidates(
            "XGBoost",
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

  def evolveBest(): XGBoostModelsWithResults = {
    evolveParameters().head
  }

  def generateScoredDataFrame(
    results: Array[XGBoostModelsWithResults]
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

  def evolveWithScoringDF(): (Array[XGBoostModelsWithResults], DataFrame) = {

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
    * @param paramsToTest Array of XGBoost Configuration (hyper parameter settings) from the post-run model
    *                     inference
    * @return The results of the hyper parameter test, as well as the scored DataFrame report.
    */
  def postRunModeledHyperParams(
    paramsToTest: Array[XGBoostConfig]
  ): (Array[XGBoostModelsWithResults], DataFrame) = {

    val finalRunResults =
      runBattery(paramsToTest, _numberOfMutationGenerations + 2)

    (finalRunResults, generateScoredDataFrame(finalRunResults))
  }

}
