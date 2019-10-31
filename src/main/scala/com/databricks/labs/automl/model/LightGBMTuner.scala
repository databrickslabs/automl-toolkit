package com.databricks.labs.automl.model

import com.databricks.labs.automl.model.tools._
import com.databricks.labs.automl.params.{
  Defaults,
  LightGBMConfig,
  LightGBMModelsWithResults
}
import com.databricks.labs.automl.utils.SparkSessionWrapper
import com.microsoft.ml.spark.lightgbm.{LightGBMClassifier, LightGBMRegressor}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.collection.parallel.mutable.ParHashSet
import scala.concurrent.forkjoin.ForkJoinPool

class LightGBMTuner(df: DataFrame, modelSelection: String, lightGBMType: String)
    extends LightGBMBase
    with SparkSessionWrapper
    with Defaults
    with Serializable
    with Evolution {

  import GBMTypes._
  import InitialGenerationMode._
  @transient private lazy val logger: Logger = Logger.getLogger(this.getClass)

  @transient private lazy val _gbmType =
    getGBMType(modelSelection, lightGBMType)

  @transient private lazy val _initialGenMode = getInitialGenMode(
    _initialGenerationMode
  )

  @transient final lazy val _uniqueLabels: Int = modelSelection match {
    case "regressor"  => 0
    case "classifier" => df.select(col(_labelCol)).distinct.count.toInt
  }

  // mutable variable instantiation

  private var _scoringMetric = _gbmType.modelType match {
    case "regressor"  => "rmse"
    case "classifier" => "f1"
    case _ =>
      throw new UnsupportedOperationException(
        s"Model $modelSelection is not supported."
      )
  }
  private var _classificationMetrics = classificationMetrics
  private var _lightgbmNumericBoundaries = _lightGBMDefaultNumBoundaries
  private var _lightgbmStringBoundaries = _lightGBMDefaultStringBoundaries

  // Setters

  def setScoringMetric(value: String): this.type = {
    _gbmType.modelType match {
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
    _scoringMetric = value
    this
  }

  /**
    * Setter for overriding the numeric boundary mappings
    * Allows for partial replacement of mappings (any not defined will use defaults)
    *
    * @param value a numeric mapping override to the defaults
    * @since 0.6.1
    * @author Ben Wilson, Databricks
    * @throws IllegalArgumentException
    */
  @throws(classOf[IllegalArgumentException])
  def setLGBMNumericBoundaries(
    value: Map[String, (Double, Double)]
  ): this.type = {
    validateNumericMapping(_lightGBMDefaultNumBoundaries, value)
    _lightgbmNumericBoundaries =
      partialOverrideNumericMapping(_lightGBMDefaultNumBoundaries, value)
    this
  }

  /**
    * Setter for partial overrides of string mappings
    *
    * @param value a string mapping override to the default values
    * @since 0.6.1
    * @author Ben Wilson, Databricks
    * @throws IllegalArgumentException
    */
  @throws(classOf[IllegalArgumentException])
  def setLGBMStringBoundaries(value: Map[String, List[String]]): this.type = {
    validateStringMapping(_lightgbmStringBoundaries, value)
    _lightgbmStringBoundaries =
      partialOverrideStringMapping(_lightgbmStringBoundaries, value)
    this
  }

  // Getters

  def getScoringMetric: String = _scoringMetric
  def getLightGBMNumericBoundaries: Map[String, (Double, Double)] =
    _lightgbmNumericBoundaries
  def getLightGBMStringBoundaries: Map[String, List[String]] =
    _lightgbmStringBoundaries
  def getClassificationMetrics: List[String] = _classificationMetrics
  def getRegressionMetrics: List[String] = regressionMetrics

  // Internal methods

  /**
    * Private internal method for resetting the metrics to employ for the scoring of each kfold during tuning and
    * evaluating model performance (primarily to select the correct type of evaluation for binary / multiclass
    * classification tasks)
    *
    * @return
    */
  private def resetClassificationMetrics: List[String] =
    _gbmType.modelType match {
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

  private def returnBestHyperParameters(
    collection: ArrayBuffer[LightGBMModelsWithResults]
  ): (LightGBMConfig, Double) = {

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
    results: ArrayBuffer[LightGBMModelsWithResults]
  ): Array[LightGBMModelsWithResults] = {
    _optimizationStrategy match {
      case "minimize" => results.result.toArray.sortWith(_.score < _.score)
      case _          => results.result.toArray.sortWith(_.score > _.score)
    }
  }

  private def sortAndReturnBestScore(
    results: ArrayBuffer[LightGBMModelsWithResults]
  ): Double = {
    sortAndReturnAll(results).head.score
  }

  private def recommendModeClassifier: String = {
    _uniqueLabels match {
      case x if x < 2 =>
        s"None. The label count [${_uniqueLabels}] is invalid for prediction."
      case x if x > 2  => s"Either gbmMulti or gbmMultiOVA"
      case x if x == 2 => s"gbmBinary"
    }
  }

  private def validateGBMClassificationSetting(): Unit = {

    _gbmType match {
      case GBMBinary =>
        if (_uniqueLabels != 2)
          throw new UnsupportedOperationException(
            s"LightGBM Model type was selected as [$lightGBMType] but the unique counts of the label column: " +
              s"[${_uniqueLabels}] is not supported by Binary Classification.  The recommended gbmModel to use is: " +
              s"$recommendModeClassifier"
          )
      case GBMMulti | GBMMultiOVA =>
        if (_uniqueLabels <= 2)
          throw new UnsupportedOperationException(
            s"LightGBM Model type was selected as [$lightGBMType] but the unique counts of the label column: " +
              s"[${_uniqueLabels}] is not supported by Multi-class Classification. The recommended gbmModel to use is: " +
              s"$recommendModeClassifier"
          )
      case _ => Unit
    }

  }

  /**
    * Private method for returning the top n parents' hyper parameters
    *
    * @param results Scored model hyper parameters results collection
    * @return the top n hyper parameters thus far
    * @since 0.6.1
    * @author Ben Wilson, Databricks
    */
  private def generateIdealParents(
    results: Array[LightGBMModelsWithResults]
  ): Array[LightGBMConfig] = {
    results.take(_numberOfParentsToRetain).map(_.modelHyperParams)
  }

  private def generateRandomThresholdedParams(
    iterationCount: Int
  ): Array[LightGBMConfig] = {

    val iterations = new ArrayBuffer[LightGBMConfig]

    var i = 0
    do {

      val baggingFraction: Double =
        generateRandomDouble("baggingFraction", _lightgbmNumericBoundaries)
      val baggingFreq: Int =
        generateRandomInteger("baggingFreq", _lightgbmNumericBoundaries)
      val featureFraction: Double =
        generateRandomDouble("featureFraction", _lightgbmNumericBoundaries)
      val learningRate: Double =
        generateRandomDouble("learningRate", _lightgbmNumericBoundaries)
      val maxBin: Int =
        generateRandomInteger("maxBin", _lightgbmNumericBoundaries)
      val maxDepth: Int =
        generateRandomInteger("maxDepth", _lightgbmNumericBoundaries)
      val minSumHessianInLeaf: Double =
        generateRandomDouble("minSumHessianInLeaf", _lightgbmNumericBoundaries)
      val numIterations: Int =
        generateRandomInteger("numIterations", _lightgbmNumericBoundaries)
      val numLeaves: Int =
        generateRandomInteger("numLeaves", _lightgbmNumericBoundaries)
      val boostFromAverage: Boolean = coinFlip()
      val lambdaL1: Double =
        generateRandomDouble("lambdaL1", _lightgbmNumericBoundaries)
      val lambdaL2: Double =
        generateRandomDouble("lambdaL2", _lightgbmNumericBoundaries)
      val alpha: Double =
        generateRandomDouble("alpha", _lightgbmNumericBoundaries)
      val boostingType: String =
        generateRandomString("boostingType", _lightgbmStringBoundaries)

      iterations += LightGBMConfig(
        baggingFraction = baggingFraction,
        baggingFreq = baggingFreq,
        featureFraction = featureFraction,
        learningRate = learningRate,
        maxBin = maxBin,
        maxDepth = maxDepth,
        minSumHessianInLeaf = minSumHessianInLeaf,
        numIterations = numIterations,
        numLeaves = numLeaves,
        boostFromAverage = boostFromAverage,
        lambdaL1 = lambdaL1,
        lambdaL2 = lambdaL2,
        alpha = alpha,
        boostingType = boostingType
      )

      i += 1
    } while (i < iterationCount)

    iterations.toArray
  }

  def generateRegressorModel(
    modelConfig: LightGBMConfig,
    gbmModelType: GBMTypes.Value
  ): LightGBMRegressor = {

    val base = new LightGBMRegressor()
      .setLabelCol(_labelCol)
      .setFeaturesCol(_featureCol)
      .setBaggingFraction(modelConfig.baggingFraction)
      .setBaggingFreq(modelConfig.baggingFreq)
      .setFeatureFraction(modelConfig.featureFraction)
      .setLearningRate(modelConfig.learningRate)
      .setMaxBin(modelConfig.maxBin)
      .setMaxDepth(modelConfig.maxDepth)
      .setMinSumHessianInLeaf(modelConfig.minSumHessianInLeaf)
      .setNumIterations(modelConfig.numIterations)
      .setNumLeaves(modelConfig.numLeaves)
      .setBoostFromAverage(modelConfig.boostFromAverage)
      .setLambdaL1(modelConfig.lambdaL1)
      .setLambdaL2(modelConfig.lambdaL2)
      .setAlpha(modelConfig.alpha)
      .setBoostingType(modelConfig.boostingType)
      .setTimeout(TIMEOUT)
      .setUseBarrierExecutionMode(BARRIER_MODE)

    gbmModelType match {
      case GBMFair     => base.setObjective("fair")
      case GBMLasso    => base.setObjective("regression_l1")
      case GBMRidge    => base.setObjective("regression_l2")
      case GBMPoisson  => base.setObjective("poisson")
      case GBMMape     => base.setObjective("mape")
      case GBMTweedie  => base.setObjective("tweedie")
      case GBMGamma    => base.setObjective("gamma")
      case GBMHuber    => base.setObjective("huber")
      case GBMQuantile => base.setObjective("quantile")
    }

  }

  def generateClassfierModel(
    modelConfig: LightGBMConfig,
    gbmModelType: GBMTypes.Value
  ): LightGBMClassifier = {

    val base = new LightGBMClassifier()
      .setLabelCol(_labelCol)
      .setFeaturesCol(_featureCol)
      .setBaggingFreq(modelConfig.baggingFreq)
      .setBaggingFraction(modelConfig.baggingFraction)
      .setFeatureFraction(modelConfig.featureFraction)
      .setLearningRate(modelConfig.learningRate)
      .setMaxBin(modelConfig.maxBin)
      .setMaxDepth(modelConfig.maxDepth)
      .setMinSumHessianInLeaf(modelConfig.minSumHessianInLeaf)
      .setNumIterations(modelConfig.numIterations)
      .setNumLeaves(modelConfig.numLeaves)
      .setBoostFromAverage(modelConfig.boostFromAverage)
      .setLambdaL1(modelConfig.lambdaL1)
      .setLambdaL2(modelConfig.lambdaL2)
      .setBoostingType(modelConfig.boostingType)
      .setTimeout(TIMEOUT)
      .setUseBarrierExecutionMode(BARRIER_MODE)

    gbmModelType match {
      case GBMBinary   => base.setObjective("binary")
      case GBMMulti    => base.setObjective("multiclass")
      case GBMMultiOVA => base.setObjective("multiclassova")
    }

  }

  /**
    * Method for performing the fit and transform with scoring for the LGBMmodel
    *
    * @param train Training data set
    * @param test Test validation data set
    * @param modelConfig configuration of hyper parameters to use
    * @param generation the generation in which the model is executing within
    * @return LightGBMModelsWithResults to store the information about the run.
    * @since 0.6.1
    * @author Ben Wilson, Databricks
    */
  def generateAndScoreGBMModel(
    train: DataFrame,
    test: DataFrame,
    modelConfig: LightGBMConfig,
    generation: Int = 1
  ): LightGBMModelsWithResults = {

    val model = _gbmType.modelType match {
      case "classifier" => generateClassfierModel(modelConfig, _gbmType)
      case _            => generateRegressorModel(modelConfig, _gbmType)
    }

    val builtModel = model.fit(train)

    val predictedData = builtModel.transform(test)

    val scoringMap = scala.collection.mutable.Map[String, Double]()

    _gbmType.modelType match {
      case "classifier" =>
        for (i <- _classificationMetrics) {
          scoringMap(i) = classificationScoring(i, _labelCol, predictedData)
        }
      case "regressor" =>
        for (i <- regressionMetrics) {
          scoringMap(i) = regressionScoring(i, _labelCol, predictedData)
        }
    }

    LightGBMModelsWithResults(
      modelConfig,
      builtModel,
      scoringMap(_scoringMetric),
      scoringMap.toMap,
      generation
    )

  }

  /**
    * Private method for execution of a collection of hyper parameters to tune against.
    * This method will instantiate models for each hyper parameter configuration, build them model, split the data,
    * train k number of models, collect the evaluated scores from each of the models, and average out the results
    * over the kFold grouping.
    *
    * @param battery Array of Configurations of the LightGBM model
    * @param generation The generation that this battery execution is operating within
    * @return Array[LightGBMModelsWithResults] that contains the results and configurations for each of the hyper
    *         parameter configurations that have been tested.
    * @since 0.6.1
    * @author Ben Wilson, Databricks
    */
  private def runBattery(
    battery: Array[LightGBMConfig],
    generation: Int = 1
  ): Array[LightGBMModelsWithResults] = {

    val metrics = _gbmType.modelType match {
      case "classifier" => _classificationMetrics
      case _            => regressionMetrics
    }

    val statusObj = new ModelReporting("lightgbm", metrics)

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    @volatile var results = ArrayBuffer[LightGBMModelsWithResults]()
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

      val kFoldBuffer = ArrayBuffer[LightGBMModelsWithResults]()

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) =
          genTestTrain(df, scala.util.Random.nextLong(), uniqueLabels)
        kFoldBuffer += generateAndScoreGBMModel(train, test, x)
      }
      val scores = ArrayBuffer[Double]()
      kFoldBuffer.map(x => { scores += x.score })

      val scoringMap = scala.collection.mutable.Map[String, Double]()

      _gbmType.modelType match {
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

      val runAvg = LightGBMModelsWithResults(
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
    parents: Array[LightGBMConfig],
    mutationCount: Int,
    mutationAggression: Int,
    mutationMagnitude: Double
  ): Array[LightGBMConfig] = {

    val mutationPayload = new ArrayBuffer[LightGBMConfig]
    val totalConfigs = modelConfigLength[LightGBMConfig]
    val indexMutation =
      if (mutationAggression >= totalConfigs) totalConfigs - 1
      else totalConfigs - mutationAggression
    val mutationCandidates = generateRandomThresholdedParams(mutationCount)
    val mutationIndeces =
      generateMutationIndeces(1, totalConfigs, indexMutation, mutationCount)

    val mutationMerge = mutationCandidates.zip(mutationIndeces)

    mutationMerge.map { x =>
      val randomParent = scala.util.Random.shuffle(parents.toList).head

      LightGBMConfig(
        if (x._2.contains(0))
          geneMixing(
            randomParent.baggingFraction,
            x._1.baggingFraction,
            mutationMagnitude
          )
        else randomParent.baggingFraction,
        if (x._2.contains(1))
          geneMixing(
            randomParent.baggingFreq,
            x._1.baggingFreq,
            mutationMagnitude
          )
        else randomParent.baggingFreq,
        if (x._2.contains(2))
          geneMixing(
            randomParent.featureFraction,
            x._1.featureFraction,
            mutationMagnitude
          )
        else randomParent.featureFraction,
        if (x._2.contains(3))
          geneMixing(
            randomParent.learningRate,
            x._1.learningRate,
            mutationMagnitude
          )
        else randomParent.learningRate,
        if (x._2.contains(4))
          geneMixing(randomParent.maxBin, x._1.maxBin, mutationMagnitude)
        else randomParent.maxBin,
        if (x._2.contains(5))
          geneMixing(randomParent.maxDepth, x._1.maxDepth, mutationMagnitude)
        else randomParent.maxDepth,
        if (x._2.contains(6))
          geneMixing(
            randomParent.minSumHessianInLeaf,
            x._1.minSumHessianInLeaf,
            mutationMagnitude
          )
        else randomParent.minSumHessianInLeaf,
        if (x._2.contains(7))
          geneMixing(
            randomParent.numIterations,
            x._1.numIterations,
            mutationMagnitude
          )
        else randomParent.numIterations,
        if (x._2.contains(8))
          geneMixing(randomParent.numLeaves, x._1.numLeaves, mutationMagnitude)
        else randomParent.numLeaves,
        if (x._2.contains(9))
          coinFlip(
            randomParent.boostFromAverage,
            x._1.boostFromAverage,
            mutationMagnitude
          )
        else randomParent.boostFromAverage,
        if (x._2.contains(10))
          geneMixing(randomParent.lambdaL1, x._1.lambdaL1, mutationMagnitude)
        else randomParent.lambdaL1,
        if (x._2.contains(11))
          geneMixing(randomParent.lambdaL2, x._1.lambdaL2, mutationMagnitude)
        else randomParent.lambdaL2,
        if (x._2.contains(12))
          geneMixing(randomParent.alpha, x._1.alpha, mutationMagnitude)
        else randomParent.alpha,
        if (x._2.contains(13))
          geneMixing(randomParent.boostingType, x._1.boostingType)
        else randomParent.boostingType
      )

    }

  }

  private def continuousEvolution(): Array[LightGBMModelsWithResults] = {

    setClassificationMetrics(resetClassificationMetrics)
    validateGBMClassificationSetting()

    val taskSupport = new ForkJoinTaskSupport(
      new ForkJoinPool(_continuousEvolutionParallelism)
    )

    var runResults = new ArrayBuffer[LightGBMModelsWithResults]

    var scoreHistory = new ArrayBuffer[Double]

    // Set the beginning of the loop and instantiate a place holder for holdling the current best score
    var iter: Int = 1
    var bestScore: Double = 0.0
    var rollingImprovement: Boolean = true
    var incrementalImprovementCount: Int = 0
    val earlyStoppingImprovementThreshold: Int =
      _continuousEvolutionImprovementThreshold

    val totalConfigs = modelConfigLength[LightGBMConfig]

    var runSet = _initialGenerationMode match {

      case "random" =>
        if (_modelSeedSet) {
          val genArray = new ArrayBuffer[LightGBMConfig]
          val startingModelSeed = generateLightGBMConfig(_modelSeed)
          genArray += startingModelSeed
          genArray ++= irradiateGeneration(
            Array(startingModelSeed),
            _firstGenerationGenePool,
            totalConfigs - 1,
            _geneticMixing
          )
          ParHashSet(genArray.result.toArray: _*)
        } else {
          ParHashSet(
            generateRandomThresholdedParams(_firstGenerationGenePool): _*
          )
        }
      case "permutations" =>
        val startingPool = new HyperParameterFullSearch()
          .setModelFamily(_gbmType.gbmType)
          .setModelType(_gbmType.modelType)
          .setPermutationCount(_initialGenerationPermutationCount)
          .setIndexMixingMode(_initialGenerationIndexMixingMode)
          .setArraySeed(_initialGenerationArraySeed)
          .initialGenerationSeedLightGBM(
            _lightgbmNumericBoundaries,
            _lightgbmStringBoundaries
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

  /**
    * Method for batch hyperparameter generational tuning.
    * @return Tuning results
    * @since 0.6.1
    * @author Ben Wilson, Databricks
    */
  private def evolveParameters(): Array[LightGBMModelsWithResults] = {

    setClassificationMetrics(resetClassificationMetrics)
    validateGBMClassificationSetting()

    var generation = 1

    val fossilRecord = ArrayBuffer[LightGBMModelsWithResults]()

    val totalConfigs = modelConfigLength[LightGBMConfig]

    val primordial = _initialGenMode match {
      case RANDOM =>
        if (_modelSeedSet) {

          val startingModelSeed = generateLightGBMConfig(_modelSeed)

          runBattery(
            irradiateGeneration(
              Array(startingModelSeed),
              _firstGenerationGenePool,
              totalConfigs - 1,
              _geneticMixing
            ) ++ Array(startingModelSeed),
            generation
          )

        } else
          runBattery(
            generateRandomThresholdedParams(_firstGenerationGenePool),
            generation
          )
    }
    fossilRecord ++= primordial
    generation += 1

    var currentIteration = 1

    if (_earlyStoppingFlag) {

      var currentBestResult = sortAndReturnBestScore(fossilRecord)

      if (evaluateStoppingScore(currentBestResult, _earlyStoppingScore)) {

        while (currentIteration <= _numberOfMutationGenerations && evaluateStoppingScore(
                 currentBestResult,
                 _earlyStoppingScore
               )) {

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

          val evolution =
            GenerationOptimizer.lightGBMCandidates(
              "LightGBM",
              _geneticMBORegressorType,
              fossilRecord,
              expandedCandidates,
              _optimizationStrategy,
              _numberOfMutationsPerGeneration
            )

          fossilRecord ++= runBattery(evolution, generation)
          generation += 1

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
          .lightGBMCandidates(
            "LightGBM",
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

  def evolveBest(): LightGBMModelsWithResults = {
    evolveParameters().head
  }

  def generateScoredDataFrame(
    results: Array[LightGBMModelsWithResults]
  ): DataFrame = {

    import spark.sqlContext.implicits._

    spark.sparkContext
      .parallelize(results.map(x => (x.generation, x.score)).toList)
      .toDF("generation", "score")
      .orderBy(col("generation").asc, col("score").asc)

  }

  def evolveWithScoringDF(): (Array[LightGBMModelsWithResults], DataFrame) = {
    val evolutionResults = _evolutionStrategy match {
      case "batch"      => evolveParameters()
      case "continuous" => continuousEvolution()
    }
    (evolutionResults, generateScoredDataFrame(evolutionResults))
  }

  def postRunModeledHyperParams(
    paramsToTest: Array[LightGBMConfig]
  ): (Array[LightGBMModelsWithResults], DataFrame) = {
    val finalRunResults =
      runBattery(paramsToTest, _numberOfMutationGenerations + 2)
    (finalRunResults, generateScoredDataFrame(finalRunResults))
  }

}

// ensure that LightGBM package is installed: com.microsoft.ml.spark:mmlspark_2.11:0.18.1
