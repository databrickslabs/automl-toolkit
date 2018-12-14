package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.params.{Defaults, GBTConfig, GBTModelsWithResults}
import com.databricks.spark.automatedml.utils.SparkSessionWrapper
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

import org.apache.log4j.{Level, Logger}

class GBTreesTuner(df: DataFrame, modelSelection: String) extends SparkSessionWrapper with Evolution with Defaults{

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _scoringMetric = modelSelection match {
    case "regressor" => "rmse"
    case "classifier" => "f1"
    case _ => throw new UnsupportedOperationException(s"Model $modelSelection is not a supported modeling mode")
  }

  private var _gbtNumericBoundaries = _gbtDefaultNumBoundaries

  private var _gbtStringBoundaries = _gbtDefaultStringBoundaries

  def setScoringMetric(value: String): this.type = {
    modelSelection match {
      case "regressor" => require(regressionMetrics.contains(value),
        s"Regressor scoring metric '$value' is not a valid member of ${
          invalidateSelection(value, regressionMetrics)
        }")
      case "classifier" => require(classificationMetrics.contains(value),
        s"Regressor scoring metric '$value' is not a valid member of ${
          invalidateSelection(value, classificationMetrics)
        }")
      case _ => throw new UnsupportedOperationException(s"Unsupported modelType $modelSelection")
    }
    this._scoringMetric = value
    this
  }

  def setGBTNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    _gbtNumericBoundaries = value
    this
  }

  def setGBTStringBoundaries(value: Map[String, List[String]]): this.type = {
    _gbtStringBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getGBTNumericBoundaries: Map[String, (Double, Double)] = _gbtNumericBoundaries

  def getGBTStringBoundaries: Map[String, List[String]] = _gbtStringBoundaries

  def getClassificationMetrics: List[String] = classificationMetrics

  def getRegressionMetrics: List[String] = regressionMetrics

  private def modelDecider[A, B](modelConfig: GBTConfig) = {

    val builtModel = modelSelection match {
      case "classifier" =>
        new GBTClassifier()
          .setLabelCol(_labelCol)
          .setFeaturesCol(_featureCol)
          .setCheckpointInterval(-1)
          .setImpurity(modelConfig.impurity)
          .setLossType(modelConfig.lossType)
          .setMaxBins(modelConfig.maxBins)
          .setMaxDepth(modelConfig.maxDepth)
          .setMaxIter(modelConfig.maxIter)
          .setMinInfoGain(modelConfig.minInfoGain)
          .setMinInstancesPerNode(modelConfig.minInstancesPerNode)
          .setStepSize(modelConfig.stepSize)
      case "regressor" =>
        new GBTRegressor()
          .setLabelCol(_labelCol)
          .setFeaturesCol(_featureCol)
          .setCheckpointInterval(-1)
          .setImpurity(modelConfig.impurity)
          .setLossType(modelConfig.lossType)
          .setMaxBins(modelConfig.maxBins)
          .setMaxDepth(modelConfig.maxDepth)
          .setMaxIter(modelConfig.maxIter)
          .setMinInfoGain(modelConfig.minInfoGain)
          .setMinInstancesPerNode(modelConfig.minInstancesPerNode)
          .setStepSize(modelConfig.stepSize)
      case _ => throw new UnsupportedOperationException(s"Unsupported modelType $modelSelection")
    }
    builtModel
  }

  override def generateRandomString(param: String, boundaryMap: Map[String, List[String]]): String = {

    val stringListing = param match {
      case "impurity" => modelSelection match {
        case "regressor" => List("variance")
        case _ => boundaryMap(param)
      }
      case "lossType" => modelSelection match {
        case "regressor" => List("squared", "absolute")
        case _ => boundaryMap(param)
      }
      case _ => boundaryMap(param)
    }
    _randomizer.shuffle(stringListing).head
  }

  private def generateThresholdedParams(iterationCount: Int): Array[GBTConfig] = {

    val iterations = new ArrayBuffer[GBTConfig]

    var i = 0
    do {
      val impurity = generateRandomString("impurity", _gbtStringBoundaries)
      val lossType = generateRandomString("lossType", _gbtStringBoundaries)
      val maxBins = generateRandomInteger("maxBins", _gbtNumericBoundaries)
      val maxDepth = generateRandomInteger("maxDepth", _gbtNumericBoundaries)
      val maxIter = generateRandomInteger("maxIter", _gbtNumericBoundaries)
      val minInfoGain = generateRandomDouble("minInfoGain", _gbtNumericBoundaries)
      val minInstancesPerNode = generateRandomInteger("minInstancesPerNode", _gbtNumericBoundaries)
      val stepSize = generateRandomDouble("stepSize", _gbtNumericBoundaries)
      iterations += GBTConfig(impurity, lossType, maxBins, maxDepth, maxIter, minInfoGain, minInstancesPerNode,
        stepSize)
      i += 1
    } while (i < iterationCount)

    iterations.toArray
  }

  private def generateAndScoreGBTModel(train: DataFrame, test: DataFrame,
                                                modelConfig: GBTConfig,
                                                generation: Int = 1): GBTModelsWithResults = {

    val gbtModel = modelDecider(modelConfig)

    val builtModel = gbtModel.fit(train)

    val predictedData = builtModel.transform(test)

    val scoringMap = scala.collection.mutable.Map[String, Double]()

    modelSelection match {
      case "classifier" =>
        for (i <- classificationMetrics) {
          val scoreEvaluator = new MulticlassClassificationEvaluator()
            .setLabelCol(_labelCol)
            .setPredictionCol("prediction")
            .setMetricName(i)
          scoringMap(i) = scoreEvaluator.evaluate(predictedData)
        }
      case "regressor" =>
        for (i <- regressionMetrics) {
          val scoreEvaluator = new RegressionEvaluator()
            .setLabelCol(_labelCol)
            .setPredictionCol("prediction")
            .setMetricName(i)
          scoringMap(i) = scoreEvaluator.evaluate(predictedData)
        }
    }

    GBTModelsWithResults(modelConfig, builtModel, scoringMap(_scoringMetric), scoringMap.toMap, generation)
  }

  private def runBattery(battery: Array[GBTConfig], generation: Int = 1): Array[GBTModelsWithResults] = {

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    @volatile var results = new ArrayBuffer[GBTModelsWithResults]
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

      val kFoldBuffer = new ArrayBuffer[GBTModelsWithResults]

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) = genTestTrain(df, scala.util.Random.nextLong)
        kFoldBuffer += generateAndScoreGBTModel(train, test, x)
      }
      val scores = new ArrayBuffer[Double]
      kFoldBuffer.map(x => {
        scores += x.score
      })

      val scoringMap = scala.collection.mutable.Map[String, Double]()
      modelSelection match {
        case "classifier" =>
          for (a <- classificationMetrics) {
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
        case _ => throw new UnsupportedOperationException(s"$modelSelection is not a supported model type.")
      }

      val runAvg = GBTModelsWithResults(x, kFoldBuffer.result.head.model, scores.sum / scores.length,
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


  private def irradiateGeneration(parents: Array[GBTConfig], mutationCount: Int,
                          mutationAggression: Int, mutationMagnitude: Double): Array[GBTConfig] = {

    val mutationPayload = new ArrayBuffer[GBTConfig]
    val totalConfigs = modelConfigLength[GBTConfig]
    val indexMutation = if (mutationAggression >= totalConfigs) totalConfigs - 1 else totalConfigs - mutationAggression
    val mutationCandidates = generateThresholdedParams(mutationCount)
    val mutationIndeces = generateMutationIndeces(1, totalConfigs, indexMutation,
      mutationCount)

    for (i <- mutationCandidates.indices) {

      val randomParent = scala.util.Random.shuffle(parents.toList).head
      val mutationIteration = mutationCandidates(i)
      val mutationIndexIteration = mutationIndeces(i)

      mutationPayload += GBTConfig(
        if (mutationIndexIteration.contains(0)) geneMixing(
          randomParent.impurity, mutationIteration.impurity)
        else randomParent.impurity,
        if (mutationIndexIteration.contains(1)) geneMixing(
          randomParent.lossType, mutationIteration.lossType)
        else randomParent.lossType,
        if (mutationIndexIteration.contains(2)) geneMixing(
          randomParent.maxBins, mutationIteration.maxBins, mutationMagnitude)
        else randomParent.maxBins,
        if (mutationIndexIteration.contains(3)) geneMixing(
          randomParent.maxDepth, mutationIteration.maxDepth, mutationMagnitude)
        else randomParent.maxDepth,
        if (mutationIndexIteration.contains(4)) geneMixing(
          randomParent.maxIter, mutationIteration.maxIter, mutationMagnitude)
        else randomParent.maxIter,
        if (mutationIndexIteration.contains(5)) geneMixing(
          randomParent.minInfoGain, mutationIteration.minInfoGain, mutationMagnitude)
        else randomParent.minInfoGain,
        if (mutationIndexIteration.contains(6)) geneMixing(
          randomParent.minInstancesPerNode, mutationIteration.minInstancesPerNode, mutationMagnitude)
        else randomParent.minInstancesPerNode,
        if (mutationIndexIteration.contains(7)) geneMixing(
          randomParent.stepSize, mutationIteration.stepSize, mutationMagnitude)
        else randomParent.stepSize
      )
    }
    mutationPayload.result.toArray
  }

  def generateIdealParents(results: Array[GBTModelsWithResults]): Array[GBTConfig] = {
    val bestParents = new ArrayBuffer[GBTConfig]
    results.take(_numberOfParentsToRetain).map(x => {
      bestParents += x.modelHyperParams
    })
    bestParents.result.toArray
  }

  def evolveParameters(startingSeed: Option[GBTConfig] = None): Array[GBTModelsWithResults] = {

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[GBTModelsWithResults]

    val totalConfigs = modelConfigLength[GBTConfig]

    val primordial = startingSeed match {
      case Some(`startingSeed`) =>
        val generativeArray = new ArrayBuffer[GBTConfig]
        generativeArray += startingSeed.asInstanceOf[GBTConfig]
        generativeArray ++= irradiateGeneration(
          Array(startingSeed.asInstanceOf[GBTConfig]),
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

  def evolveBest(startingSeed: Option[GBTConfig] = None): GBTModelsWithResults = {
    evolveParameters(startingSeed).head
  }

  def generateScoredDataFrame(results: Array[GBTModelsWithResults]): DataFrame = {

    import spark.sqlContext.implicits._

    val scoreBuffer = new ListBuffer[(Int, Double)]
    results.map(x => scoreBuffer += ((x.generation, x.score)))
    val scored = scoreBuffer.result
    spark.sparkContext.parallelize(scored)
      .toDF("generation", "score").orderBy(col("generation").asc, col("score").asc)
  }

  def evolveWithScoringDF(startingSeed: Option[GBTConfig] = None): (Array[GBTModelsWithResults], DataFrame) = {
    val evolutionResults = evolveParameters(startingSeed)
    (evolutionResults, generateScoredDataFrame(evolutionResults))
  }

}
