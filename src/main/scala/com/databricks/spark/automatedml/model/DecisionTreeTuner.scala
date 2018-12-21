package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.params.{Defaults, TreesConfig, TreesModelsWithResults}
import com.databricks.spark.automatedml.utils.SparkSessionWrapper
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.functions.col

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

import org.apache.log4j.{Level, Logger}

class DecisionTreeTuner(df: DataFrame, modelSelection: String) extends SparkSessionWrapper with Evolution with
  Defaults {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _scoringMetric = modelSelection match {
    case "regressor" => "rmse"
    case "classifier" => "f1"
    case _ => throw new UnsupportedOperationException(s"Model $modelSelection is not supported.")
  }

  private var _treesNumericBoundaries = _treesDefaultNumBoundaries

  private var _treesStringBoundaries = _treesDefaultStringBoundaries

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

  def setTreesNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    _treesNumericBoundaries = value
    this
  }

  def setTreesStringBoundaries(value: Map[String, List[String]]): this.type = {
    _treesStringBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getTreesNumericBoundaries: Map[String, (Double, Double)] = _treesNumericBoundaries

  def getTreesStringBoundaries: Map[String, List[String]] = _treesStringBoundaries

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
      case _ => throw new UnsupportedOperationException(s"Unsupported model type $modelSelection")
    }
    builtModel
  }

  override def generateRandomString(param: String, boundaryMap: Map[String, List[String]]): String = {

    val stringListing = param match {
      case "impurity" => modelSelection match {
        case "regressor" => List("variance")
        case _ => boundaryMap(param)
      }
      case _ => boundaryMap(param)
    }
    _randomizer.shuffle(stringListing).head
  }

  private def generateThresholdedParams(iterationCount: Int): Array[TreesConfig] = {

    val iterations = new ArrayBuffer[TreesConfig]

    var i = 0
    do {
      val impurity = generateRandomString("impurity", _treesStringBoundaries)
      val maxBins = generateRandomInteger("maxBins", _treesNumericBoundaries)
      val maxDepth = generateRandomInteger("maxDepth", _treesNumericBoundaries)
      val minInfoGain = generateRandomDouble("minInfoGain", _treesNumericBoundaries)
      val minInstancesPerNode = generateRandomInteger("minInstancesPerNode", _treesNumericBoundaries)

      iterations += TreesConfig(impurity, maxBins, maxDepth, minInfoGain, minInstancesPerNode)
      i += 1
    } while (i < iterationCount)

    iterations.toArray
  }

  private def generateAndScoreTreesModel(train: DataFrame, test: DataFrame,
                                                modelConfig: TreesConfig,
                                                generation: Int = 1): TreesModelsWithResults = {

    val treesModel = modelDecider(modelConfig)

    val builtModel = treesModel.fit(train)

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

    TreesModelsWithResults(modelConfig, builtModel, scoringMap(_scoringMetric), scoringMap.toMap, generation)
  }

  private def runBattery(battery: Array[TreesConfig], generation: Int = 1): Array[TreesModelsWithResults] = {

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    @volatile var results = new ArrayBuffer[TreesModelsWithResults]
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

      val kFoldBuffer = new ArrayBuffer[TreesModelsWithResults]

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) = genTestTrain(df, scala.util.Random.nextLong)
        kFoldBuffer += generateAndScoreTreesModel(train, test, x)
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

      val runAvg = TreesModelsWithResults(x, kFoldBuffer.result.head.model, scores.sum / scores.length,
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

  private def irradiateGeneration(parents: Array[TreesConfig], mutationCount: Int,
                                  mutationAggression: Int, mutationMagnitude: Double): Array[TreesConfig] = {

    val mutationPayload = new ArrayBuffer[TreesConfig]
    val totalConfigs = modelConfigLength[TreesConfig]
    val indexMutation = if (mutationAggression >= totalConfigs) totalConfigs - 1 else totalConfigs - mutationAggression
    val mutationCandidates = generateThresholdedParams(mutationCount)
    val mutationIndeces = generateMutationIndeces(1, totalConfigs, indexMutation,
      mutationCount)

    for (i <- mutationCandidates.indices) {

      val randomParent = scala.util.Random.shuffle(parents.toList).head
      val mutationIteration = mutationCandidates(i)
      val mutationIndexIteration = mutationIndeces(i)

      mutationPayload += TreesConfig(
        if (mutationIndexIteration.contains(0)) geneMixing(
          randomParent.impurity, mutationIteration.impurity)
        else randomParent.impurity,
        if (mutationIndexIteration.contains(1)) geneMixing(
          randomParent.maxBins, mutationIteration.maxBins, mutationMagnitude)
        else randomParent.maxBins,
        if (mutationIndexIteration.contains(2)) geneMixing(
          randomParent.maxDepth, mutationIteration.maxDepth, mutationMagnitude)
        else randomParent.maxDepth,
        if (mutationIndexIteration.contains(3)) geneMixing(
          randomParent.minInfoGain, mutationIteration.minInfoGain, mutationMagnitude)
        else randomParent.minInfoGain,
        if (mutationIndexIteration.contains(4)) geneMixing(
          randomParent.minInstancesPerNode, mutationIteration.minInstancesPerNode, mutationMagnitude)
        else randomParent.minInstancesPerNode
      )
    }
    mutationPayload.result.toArray
  }

  def generateIdealParents(results: Array[TreesModelsWithResults]): Array[TreesConfig] = {
    val bestParents = new ArrayBuffer[TreesConfig]
    results.take(_numberOfParentsToRetain).map(x => {
      bestParents += x.modelHyperParams
    })
    bestParents.result.toArray
  }

  def evolveParameters(startingSeed: Option[TreesConfig] = None): Array[TreesModelsWithResults] = {

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[TreesModelsWithResults]

    val totalConfigs = modelConfigLength[TreesConfig]

    val primordial = startingSeed match {
      case Some(`startingSeed`) =>
        val generativeArray = new ArrayBuffer[TreesConfig]
        generativeArray += startingSeed.asInstanceOf[TreesConfig]
        generativeArray ++= irradiateGeneration(
          Array(startingSeed.asInstanceOf[TreesConfig]),
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

  def evolveBest(startingSeed: Option[TreesConfig] = None): TreesModelsWithResults = {
    evolveParameters(startingSeed).head
  }

  def generateScoredDataFrame(results: Array[TreesModelsWithResults]): DataFrame = {

    import spark.sqlContext.implicits._

    val scoreBuffer = new ListBuffer[(Int, Double)]
    results.map(x => scoreBuffer += ((x.generation, x.score)))
    val scored = scoreBuffer.result
    spark.sparkContext.parallelize(scored)
      .toDF("generation", "score").orderBy(col("generation").asc, col("score").asc)
  }

  def evolveWithScoringDF(startingSeed: Option[TreesConfig] = None):
  (Array[TreesModelsWithResults], DataFrame) = {
    val evolutionResults = evolveParameters(startingSeed)
    (evolutionResults, generateScoredDataFrame(evolutionResults))
  }

}
