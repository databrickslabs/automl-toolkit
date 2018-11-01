package com.databricks.spark.automatedml

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.reflect.runtime.universe._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}

//TODO: change the par mapping to proper thread pooling?
//TODO: feature flag for logging to MLFlow, retain all the scoring and metrics.

class GeneticTuner(df: DataFrame, modelSelection: String) extends DataValidation with SparkSessionWrapper {

  private var _labelCol = "label"
  private var _featuresCol = "features"
  private var _trainPortion = 0.8
  private var _kFold = 3
  private var _seed = 42L
  private var _scoringMetric = modelSelection match {
    case "regressor" => "rmse"
    case "classifier" => "f1"
    case _ => throw new UnsupportedOperationException(s"Model $modelSelection is not supported.")
  }
  private var _optimizationStrategy = "maximize"
  private var _firstGenerationGenePool = 20
  private var _numberOfMutationGenerations = 10
  private var _numberOfParentsToRetain = 3
  private var _numberOfMutationsPerGeneration = 10
  private var _geneticMixing = 0.7
  private var _generationalMutationStrategy = "linear"
  private var _mutationMagnitudeMode = "random"
  private var _fixedMutationValue = 1

  private var _kFoldIteratorRange = Range(0, _kFold).par

  final val allowableStrategies = Seq("minimize", "maximize")
  final val allowableMutationStrategies = Seq("linear", "fixed")
  final val allowableMutationMagnitudeMode = Seq("random", "fixed")

  final val classificationMetrics = List("f1", "weightedPrecision", "weightedRecall", "accuracy")
  final val regressionMetrics = List("rmse", "mse", "r2", "mae")

  private val _randomizer = scala.util.Random
  _randomizer.setSeed(_seed)

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

  def setLabelCol(value: String): this.type = {
    this._labelCol = value
    this
  }

  def setFeaturesCol(value: String): this.type = {
    this._featuresCol = value
    this
  }

  def setTrainPortion(value: Double): this.type = {
    assert(value < 1.0 & value > 0.0, "Training portion must be in the range > 0 and < 1")
    this._trainPortion = value
    this
  }

  def setKFold(value: Int): this.type = {
    this._kFold = value
    this._kFoldIteratorRange = Range(0, _kFold).par
    this
  }

  def setSeed(value: Long): this.type = {
    this._seed = value
    this
  }

  def setScoringMetric(value: String): this.type = {
    modelSelection match {
      case "regressor" => assert(regressionMetrics.contains(value),
        s"Regressor scoring optimization '$value' is not a valid member of ${
          invalidateSelection(value, regressionMetrics)}")
      case "classifier" => assert(classificationMetrics.contains(value),
        s"Regressor scoring optimization '$value' is not a valid member of ${
          invalidateSelection(value, classificationMetrics)}")
      case _ => throw new UnsupportedOperationException(s"Unsupported modelType $modelSelection")
    }
    this._scoringMetric = value
    this
  }

  def setOptimizationStrategy(value: String): this.type = {
    val valueLC = value.toLowerCase
    assert(allowableStrategies.contains(valueLC),
      s"Optimization Strategy '$valueLC' is not a member of ${
        invalidateSelection(valueLC, allowableStrategies)}")
    this._optimizationStrategy = valueLC
    this
  }

  def setFirstGenerationGenePool(value: Int): this.type = {
    assert(value > 5, s"Values less than 5 for firstGenerationGenePool will require excessive generational mutation to converge")
    this._firstGenerationGenePool = value
    this
  }

  def setNumberOfMutationGenerations(value: Int): this.type = {
    assert(value > 0, s"Number of Generations must be greater than 0")
    this._numberOfMutationGenerations = value
    this
  }

  def setNumberOfParentsToRetain(value: Int): this.type = {
    assert(value > 0, s"Number of Parents must be greater than 0. '$value' is not a valid number.")
    this._numberOfParentsToRetain = value
    this
  }

  def setNumberOfMutationsPerGeneration(value: Int): this.type = {
    assert(value > 0, s"Number of Mutations per generation must be greater than 0. '$value' is not a valid number.")
    this._numberOfMutationsPerGeneration = value
    this
  }

  def setGeneticMixing(value: Double): this.type = {
    assert(value < 1.0 & value > 0.0,
      s"Mutation Aggressiveness must be in range (0,1). Current Setting of $value is not permitted.")
    this._geneticMixing = value
    this
  }

  def setGenerationalMutationStrategy(value: String): this.type = {
    val valueLC = value.toLowerCase
    assert(allowableMutationStrategies.contains(valueLC),
      s"Generational Mutation Strategy '$valueLC' is not a member of ${
        invalidateSelection(valueLC, allowableMutationStrategies)}")
    this._generationalMutationStrategy = valueLC
    this
  }

  def setMutationMagnitudeMode(value: String): this.type = {
    val valueLC = value.toLowerCase
    assert(allowableMutationMagnitudeMode.contains(valueLC),
      s"Mutation Magnitude Mode '$valueLC' is not a member of ${
        invalidateSelection(valueLC, allowableMutationMagnitudeMode)}")
    this._mutationMagnitudeMode = valueLC
    this
  }

  def setFixedMutationValue(value: Int): this.type = {
    val maxMutationCount = modelConfigLength[RandomForestConfig]
    assert(value <= maxMutationCount,
      s"Mutation count '$value' cannot exceed number of hyperparameters ($maxMutationCount)")
    assert(value > 0, s"Mutation count '$value' must be greater than 0")
    this._fixedMutationValue = value
    this
  }

  def setRandomForestNumericBoundaries(value: Map[String,(Double, Double)]): this.type = {
    this._randomForestNumericBoundaries = value
    this
  }

  def setRandomForestStringBoundaries(value: Map[String, List[String]]): this.type = {
    this._randomForestStringBoundaries = value
    this
  }

  def getLabelCol: String = _labelCol
  def getFeaturesCol: String = _featuresCol
  def getTrainPortion: Double = _trainPortion
  def getKFold: Int = _kFold
  def getSeed: Long = _seed
  def getScoringMetric: String = _scoringMetric
  def getOptimizationStrategy: String = _optimizationStrategy
  def getFirstGenerationGenePool: Int = _firstGenerationGenePool
  def getNumberOfMutationGenerations: Int = _numberOfMutationGenerations
  def getNumberOfParentsToRetain: Int = _numberOfParentsToRetain
  def getNumberOfMutationsPerGeneration: Int = _numberOfMutationsPerGeneration
  def getGeneticMixing: Double = _geneticMixing
  def getGenerationalMutationStrategy: String = _generationalMutationStrategy
  def getMutationMagnitudeMode: String = _mutationMagnitudeMode
  def getFixedMutationValue: Int = _fixedMutationValue
  def getRandomForestNumericBoundaries: Map[String, (Double, Double)] = _randomForestNumericBoundaries
  def getRandomForestStringBoundaries: Map[String, List[String]] = _randomForestStringBoundaries

  private def modelConfigLength[T: TypeTag]: Int = {
    typeOf[T].members.collect{
      case m: MethodSymbol if m.isCaseAccessor => m
    }.toList.length
  }

  private def genTestTrain(data: DataFrame, seed: Long): Array[DataFrame] = {
    data.randomSplit(Array(_trainPortion, 1-_trainPortion), seed)
  }

  private def modelDecider[A,B](modelConfig: RandomForestConfig) = {

    val builtModel = modelSelection match {
      case "classifier" =>
        new RandomForestClassifier()
          .setLabelCol(_labelCol)
          .setFeaturesCol(_featuresCol)
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
          .setFeaturesCol(_featuresCol)
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

  private def extractBoundaryDouble(param: String, boundaryMap: Map[String, (AnyVal, AnyVal)]): (Double, Double) = {
    val minimum = boundaryMap(param)._1.asInstanceOf[Double]
    val maximum = boundaryMap(param)._2.asInstanceOf[Double]
    (minimum, maximum)
  }

  private def extractBoundaryInteger(param: String, boundaryMap: Map[String, (AnyVal, AnyVal)]): (Int, Int) = {
    val minimum = boundaryMap(param)._1.asInstanceOf[Double].toInt
    val maximum = boundaryMap(param)._2.asInstanceOf[Double].toInt
    (minimum, maximum)
  }

  private def generateRandomDouble(param: String, boundaryMap: Map[String, (AnyVal, AnyVal)]): Double = {
    val (minimumValue, maximumValue) = extractBoundaryDouble(param, boundaryMap)
    (_randomizer.nextDouble * (maximumValue - minimumValue)) + minimumValue
  }

  private def generateRandomInteger(param: String, boundaryMap: Map[String, (AnyVal, AnyVal)]): Int = {
    val (minimumValue, maximumValue) = extractBoundaryInteger(param, boundaryMap)
    _randomizer.nextInt(maximumValue - minimumValue) + minimumValue
  }

  private def generateRandomString(param: String, boundaryMap: Map[String, List[String]]): String = {

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
    }while(i < iterationCount)

    iterations.toArray
  }

  private def generateAndScoreRandomForestModel(train: DataFrame, test: DataFrame,
                                                modelConfig: RandomForestConfig, generation: Int=1): ModelsWithResults = {

    val randomForestModel = modelDecider(modelConfig)

    val builtModel = randomForestModel.fit(train)

    val predictedData = builtModel.transform(test)

    val scoringMap = scala.collection.mutable.Map[String, Double]()

    modelSelection match {
      case "classifier" =>
        for(i <- classificationMetrics) {
          val scoreEvaluator = new MulticlassClassificationEvaluator()
            .setLabelCol(_labelCol)
            .setPredictionCol("prediction")
            .setMetricName(i)
          scoringMap(i) = scoreEvaluator.evaluate(predictedData)
        }
      case "regressor" =>
        for(i <- regressionMetrics) {
          val scoreEvaluator = new RegressionEvaluator()
            .setLabelCol(_labelCol)
            .setPredictionCol("prediction")
            .setMetricName(i)
          scoringMap(i) = scoreEvaluator.evaluate(predictedData)
        }
    }

    ModelsWithResults(modelConfig, builtModel, scoringMap(_scoringMetric), scoringMap.toMap, generation)
  }


  def runBattery(battery: Array[RandomForestConfig], generation: Int=1): Array[ModelsWithResults] = {

    val dfSchema = df.schema
    assert(dfSchema.fieldNames.contains(_labelCol),
      s"Dataframe does not contain label column named: ${_labelCol}")
    assert(dfSchema.fieldNames.contains(_featuresCol),
      s"Dataframe does not contain features column named: ${_featuresCol}")

    val results = new ArrayBuffer[ModelsWithResults]
    val runs = battery.par

    runs.foreach{x =>

      val kFoldBuffer = new ArrayBuffer[ModelsWithResults]

      for (_ <- _kFoldIteratorRange) {
        val Array(train, test) = genTestTrain(df, scala.util.Random.nextLong)
        kFoldBuffer += generateAndScoreRandomForestModel(train, test, x)
      }
      val scores = new ArrayBuffer[Double]
      kFoldBuffer.map(x => {scores += x.score})

      val scoringMap = scala.collection.mutable.Map[String, Double]()
      modelSelection match {
        case "classifier" =>
          for(a <- classificationMetrics){
            val metricScores = new ListBuffer[Double]
            kFoldBuffer.map(x => metricScores += x.evalMetrics(a))
            scoringMap(a) = metricScores.sum/metricScores.length
          }
        case "regressor" =>
          for(a <- regressionMetrics){
            val metricScores = new ListBuffer[Double]
            kFoldBuffer.map(x => metricScores += x.evalMetrics(a))
            scoringMap(a) = metricScores.sum/metricScores.length
          }
        case _ => throw new UnsupportedOperationException(s"$modelSelection is not a supported model type.")
      }


      val runAvg = ModelsWithResults(x, kFoldBuffer.result.head.model, scores.sum/scores.length,
        scoringMap.toMap, generation)
      results += runAvg
    }
    _optimizationStrategy match {
      case "minimize" => results.toArray.sortWith(_.score < _.score)
      case _ => results.toArray.sortWith(_.score > _.score)
    }
  }

  private def getRandomIndeces(minimum: Int, maximum: Int, parameterCount: Int): List[Int] = {
    val fullIndexArray = List.range(0, maximum)
    val randomSeed = new scala.util.Random
    val count = minimum + randomSeed.nextInt((parameterCount - minimum) + 1)
    val adjCount = if(count < 1) 1 else count
    val shuffledArray = scala.util.Random.shuffle(fullIndexArray).take(adjCount)
    shuffledArray.sortWith(_<_)
  }

  private def getFixedIndeces(minimum: Int, maximum: Int, parameterCount: Int): List[Int] = {
    val fullIndexArray = List.range(0, maximum)
    val randomSeed = new scala.util.Random
    randomSeed.shuffle(fullIndexArray).take(parameterCount).sortWith(_<_)
  }

  private def generateMutationIndeces(minimum: Int, maximum: Int, parameterCount: Int,
                                      mutationCount: Int): Array[List[Int]] = {
    val mutations = new ArrayBuffer[List[Int]]
    for (_ <- 0 to mutationCount){
      _mutationMagnitudeMode match {
        case "random" => mutations += getRandomIndeces(minimum, maximum, parameterCount)
        case "fixed" => mutations += getFixedIndeces(minimum, maximum, parameterCount)
        case _ => new UnsupportedOperationException(s"Unsupported mutationMagnitudeMode ${_mutationMagnitudeMode}")
      }
    }
    mutations.result.toArray
  }

  private def generateIdealParents(results: Array[ModelsWithResults]): Array[RandomForestConfig] = {
    val bestParents = new ArrayBuffer[RandomForestConfig]
    results.take(_numberOfParentsToRetain).map(x => {
      bestParents += x.modelHyperParams
    })
    bestParents.result.toArray
  }

  private def geneMixing(parent: Double, child: Double, parentMutationPercentage: Double): Double = {
    (parent * parentMutationPercentage) + (child * (1-parentMutationPercentage))
  }

  private def geneMixing(parent: Int, child: Int, parentMutationPercentage: Double): Int= {
    ((parent * parentMutationPercentage) + (child * (1-parentMutationPercentage))).toInt
  }

  private def geneMixing(parent: String, child: String): String = {
    val mixed = new ArrayBuffer[String]
    mixed += parent += child
    scala.util.Random.shuffle(mixed.toList).head
  }

  def irradiateGeneration(parents: Array[RandomForestConfig], mutationCount: Int,
                          mutationAggression: Int, mutationMagnitude: Double): Array[RandomForestConfig] = {

    val mutationPayload = new ArrayBuffer[RandomForestConfig]
    val totalConfigs = modelConfigLength[RandomForestConfig]
    val indexMutation = if(mutationAggression >= totalConfigs) totalConfigs - 1 else totalConfigs-mutationAggression
    val mutationCandidates = generateThresholdedParams(mutationCount)
    val mutationIndeces = generateMutationIndeces(1, totalConfigs, indexMutation,
      mutationCount)

    for(i <- mutationCandidates.indices) {

      val randomParent = scala.util.Random.shuffle(parents.toList).head
      val mutationIteration = mutationCandidates(i)
      val mutationIndexIteration = mutationIndeces(i)

      mutationPayload += RandomForestConfig(
        if(mutationIndexIteration.contains(0)) geneMixing(
          randomParent.numTrees, mutationIteration.numTrees, mutationMagnitude)
        else randomParent.numTrees,
        if(mutationIndexIteration.contains(1)) geneMixing(
          randomParent.impurity, mutationIteration.impurity)
        else randomParent.impurity,
        if(mutationIndexIteration.contains(2)) geneMixing(
          randomParent.maxBins, mutationIteration.maxBins, mutationMagnitude)
        else randomParent.maxBins,
        if(mutationIndexIteration.contains(3)) geneMixing(
          randomParent.maxDepth, mutationIteration.maxDepth, mutationMagnitude)
        else randomParent.maxDepth,
        if(mutationIndexIteration.contains(4)) geneMixing(
          randomParent.minInfoGain, mutationIteration.minInfoGain, mutationMagnitude)
        else randomParent.minInfoGain,
        if(mutationIndexIteration.contains(5)) geneMixing(
          randomParent.subSamplingRate, mutationIteration.subSamplingRate, mutationMagnitude)
        else randomParent.subSamplingRate,
        if(mutationIndexIteration.contains(6)) geneMixing(
          randomParent.featureSubsetStrategy, mutationIteration.featureSubsetStrategy)
        else randomParent.featureSubsetStrategy
      )
    }
    mutationPayload.result.toArray
  }

  def evolveParameters(startingSeed: Option[RandomForestConfig]=None): Array[ModelsWithResults] = {

    var generation = 1
    // Record of all generations results
    val fossilRecord = new ArrayBuffer[ModelsWithResults]

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
        case "linear" => if(totalConfigs - (i + 1) < 1) 1 else totalConfigs - (i + 1)
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

  def evolveBest(startingSeed: Option[RandomForestConfig] = None): ModelsWithResults = {
    evolveParameters(startingSeed).head
  }

  def generateScoredDataFrame(results: Array[ModelsWithResults]): DataFrame = {

    import spark.sqlContext.implicits._

    val scoreBuffer = new ListBuffer[(Int, Double)]
    results.map(x => scoreBuffer += ((x.generation, x.score)))
    val scored = scoreBuffer.result
    spark.sparkContext.parallelize(scored)
      .toDF("generation", "score").orderBy(col("generation").asc, col("score").asc)
  }

  def evolveWithScoringDF(startingSeed: Option[RandomForestConfig]=None):
  (Array[ModelsWithResults], DataFrame) = {
    val evolutionResults = evolveParameters(startingSeed)
    (evolutionResults, generateScoredDataFrame(evolutionResults))
  }

}

