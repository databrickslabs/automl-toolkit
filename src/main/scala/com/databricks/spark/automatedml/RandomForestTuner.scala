package com.databricks.spark.automatedml

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

//TODO: change the par mapping to proper thread pooling?
//TODO: feature flag for logging to MLFlow, retain all the scoring and metrics.

class RandomForestTuner(df: DataFrame, modelSelection: String) extends SparkSessionWrapper
  with Evolution {

  private var _scoringMetric = modelSelection match {
    case "regressor" => "rmse"
    case "classifier" => "f1"
    case _ => throw new UnsupportedOperationException(s"Model $modelSelection is not supported.")
  }

  private var _randomForestNumericBoundaries = Map(
    "numTrees" -> Tuple2(50.0, 1000.0),
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "subSamplingRate" -> Tuple2(0.5, 1.0)
  )

  private var _randomForestStringBoundaries = Map(
    "impurity" -> List("gini", "entropy"),
    "featureSubsetStrategy" -> List("all", "sqrt", "log2", "onethird")
  )

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

  def setRandomForestNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    this._randomForestNumericBoundaries = value
    this
  }

  def setRandomForestStringBoundaries(value: Map[String, List[String]]): this.type = {
    this._randomForestStringBoundaries = value
    this
  }

  def getScoringMetric: String = _scoringMetric

  def getRandomForestNumericBoundaries: Map[String, (Double, Double)] = _randomForestNumericBoundaries

  def getRandomForestStringBoundaries: Map[String, List[String]] = _randomForestStringBoundaries

  def getClassificationMetrics: List[String] = classificationMetrics

  def getRegressionMetrics: List[String] = regressionMetrics

  private def modelDecider[A, B](modelConfig: RandomForestConfig) = {

    val builtModel = modelSelection match {
      case "classifier" =>
        new RandomForestClassifier()
          .setLabelCol(_labelCol)
          .setFeaturesCol(_featureCol)
          .setNumTrees(modelConfig.numTrees)
          .setCheckpointInterval(-1)
          .setImpurity(modelConfig.impurity)
          .setMaxBins(modelConfig.maxBins)
          .setMaxDepth(modelConfig.maxDepth)
          .setMinInfoGain(modelConfig.minInfoGain)
          .setFeatureSubsetStrategy(modelConfig.featureSubsetStrategy)
          .setSubsamplingRate(modelConfig.subSamplingRate)
      case "regressor" =>
        new RandomForestRegressor()
          .setLabelCol(_labelCol)
          .setFeaturesCol(_featureCol)
          .setNumTrees(modelConfig.numTrees)
          .setCheckpointInterval(-1)
          .setImpurity(modelConfig.impurity)
          .setMaxBins(modelConfig.maxBins)
          .setMaxDepth(modelConfig.maxDepth)
          .setMinInfoGain(modelConfig.minInfoGain)
          .setFeatureSubsetStrategy(modelConfig.featureSubsetStrategy)
          .setSubsamplingRate(modelConfig.subSamplingRate)
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
      case _ => boundaryMap(param)
    }
    _randomizer.shuffle(stringListing).head
  }

  private def generateThresholdedParams(iterationCount: Int): Array[RandomForestConfig] = {

    val iterations = new ArrayBuffer[RandomForestConfig]

    var i = 0
    do {
      val featureSubsetStrategy = generateRandomString("featureSubsetStrategy", _randomForestStringBoundaries)
      val subSamplingRate = generateRandomDouble("subSamplingRate", _randomForestNumericBoundaries)
      val impurity = generateRandomString("impurity", _randomForestStringBoundaries)
      val minInfoGain = generateRandomDouble("minInfoGain", _randomForestNumericBoundaries)
      val maxBins = generateRandomInteger("maxBins", _randomForestNumericBoundaries)
      val numTrees = generateRandomInteger("numTrees", _randomForestNumericBoundaries)
      val maxDepth = generateRandomInteger("maxDepth", _randomForestNumericBoundaries)
      iterations += RandomForestConfig(numTrees, impurity, maxBins, maxDepth, minInfoGain, subSamplingRate,
        featureSubsetStrategy)
      i += 1
    } while (i < iterationCount)

    iterations.toArray
  }

  private def generateAndScoreRandomForestModel(train: DataFrame, test: DataFrame,
                                                modelConfig: RandomForestConfig,
                                                generation: Int = 1): RandomForestModelsWithResults = {

    val randomForestModel = modelDecider(modelConfig)

    val builtModel = randomForestModel.fit(train)

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

    RandomForestModelsWithResults(modelConfig, builtModel, scoringMap(_scoringMetric), scoringMap.toMap, generation)
  }


  def runBattery(battery: Array[RandomForestConfig], generation: Int = 1): Array[RandomForestModelsWithResults] = {

    validateLabelAndFeatures(df, _labelCol, _featureCol)

    val results = new ArrayBuffer[RandomForestModelsWithResults]
    val runs = battery.par

    runs.foreach { x =>

      val kFoldBuffer = new ArrayBuffer[RandomForestModelsWithResults]

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) = genTestTrain(df, scala.util.Random.nextLong)
        kFoldBuffer += generateAndScoreRandomForestModel(train, test, x)
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


      val runAvg = RandomForestModelsWithResults(x, kFoldBuffer.result.head.model, scores.sum / scores.length,
        scoringMap.toMap, generation)
      results += runAvg
    }
    _optimizationStrategy match {
      case "minimize" => results.toArray.sortWith(_.score < _.score)
      case _ => results.toArray.sortWith(_.score > _.score)
    }
  }

  def irradiateGeneration(parents: Array[RandomForestConfig], mutationCount: Int,
                          mutationAggression: Int, mutationMagnitude: Double): Array[RandomForestConfig] = {

    val mutationPayload = new ArrayBuffer[RandomForestConfig]
    val totalConfigs = modelConfigLength[RandomForestConfig]
    val indexMutation = if (mutationAggression >= totalConfigs) totalConfigs - 1 else totalConfigs - mutationAggression
    val mutationCandidates = generateThresholdedParams(mutationCount)
    val mutationIndeces = generateMutationIndeces(1, totalConfigs, indexMutation,
      mutationCount)

    for (i <- mutationCandidates.indices) {

      val randomParent = scala.util.Random.shuffle(parents.toList).head
      val mutationIteration = mutationCandidates(i)
      val mutationIndexIteration = mutationIndeces(i)

      mutationPayload += RandomForestConfig(
        if (mutationIndexIteration.contains(0)) geneMixing(
          randomParent.numTrees, mutationIteration.numTrees, mutationMagnitude)
        else randomParent.numTrees,
        if (mutationIndexIteration.contains(1)) geneMixing(
          randomParent.impurity, mutationIteration.impurity)
        else randomParent.impurity,
        if (mutationIndexIteration.contains(2)) geneMixing(
          randomParent.maxBins, mutationIteration.maxBins, mutationMagnitude)
        else randomParent.maxBins,
        if (mutationIndexIteration.contains(3)) geneMixing(
          randomParent.maxDepth, mutationIteration.maxDepth, mutationMagnitude)
        else randomParent.maxDepth,
        if (mutationIndexIteration.contains(4)) geneMixing(
          randomParent.minInfoGain, mutationIteration.minInfoGain, mutationMagnitude)
        else randomParent.minInfoGain,
        if (mutationIndexIteration.contains(5)) geneMixing(
          randomParent.subSamplingRate, mutationIteration.subSamplingRate, mutationMagnitude)
        else randomParent.subSamplingRate,
        if (mutationIndexIteration.contains(6)) geneMixing(
          randomParent.featureSubsetStrategy, mutationIteration.featureSubsetStrategy)
        else randomParent.featureSubsetStrategy
      )
    }
    mutationPayload.result.toArray
  }

  def generateIdealParents(results: Array[RandomForestModelsWithResults]): Array[RandomForestConfig] = {
    val bestParents = new ArrayBuffer[RandomForestConfig]
    results.take(_numberOfParentsToRetain).map(x => {
      bestParents += x.modelHyperParams
    })
    bestParents.result.toArray
  }

  def evolveParameters(startingSeed: Option[RandomForestConfig] = None): Array[RandomForestModelsWithResults] = {

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[RandomForestModelsWithResults]

    val totalConfigs = modelConfigLength[RandomForestConfig]

    val primordial = startingSeed match {
      case Some(`startingSeed`) =>
        val generativeArray = new ArrayBuffer[RandomForestConfig]
        generativeArray += startingSeed.asInstanceOf[RandomForestConfig]
        generativeArray ++= irradiateGeneration(
          Array(startingSeed.asInstanceOf[RandomForestConfig]),
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

  def evolveBest(startingSeed: Option[RandomForestConfig] = None): RandomForestModelsWithResults = {
    evolveParameters(startingSeed).head
  }

  def generateScoredDataFrame(results: Array[RandomForestModelsWithResults]): DataFrame = {

    import spark.sqlContext.implicits._

    val scoreBuffer = new ListBuffer[(Int, Double)]
    results.map(x => scoreBuffer += ((x.generation, x.score)))
    val scored = scoreBuffer.result
    spark.sparkContext.parallelize(scored)
      .toDF("generation", "score").orderBy(col("generation").asc, col("score").asc)
  }

  def evolveWithScoringDF(startingSeed: Option[RandomForestConfig] = None):
  (Array[RandomForestModelsWithResults], DataFrame) = {
    val evolutionResults = evolveParameters(startingSeed)
    (evolutionResults, generateScoredDataFrame(evolutionResults))
  }

}

