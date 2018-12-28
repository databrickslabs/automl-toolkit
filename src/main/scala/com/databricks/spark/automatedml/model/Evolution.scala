package com.databricks.spark.automatedml.model

import com.databricks.spark.automatedml.params.RandomForestConfig
import com.databricks.spark.automatedml.utils.DataValidation
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe._

trait Evolution extends DataValidation {

  // TODO: move all / most of these over to configuration defaults?
  var _labelCol = "label"
  var _featureCol = "features"
  var _trainPortion = 0.8
  var _parallelism = 20
  var _kFold = 3
  var _seed = 42L
  var _kFoldIteratorRange: scala.collection.parallel.immutable.ParRange = Range(0, _kFold).par

  var _optimizationStrategy = "maximize"
  var _firstGenerationGenePool = 20
  var _numberOfMutationGenerations = 10
  var _numberOfParentsToRetain = 3
  var _numberOfMutationsPerGeneration = 10
  var _geneticMixing = 0.7
  var _generationalMutationStrategy = "linear"
  var _mutationMagnitudeMode = "random"
  var _fixedMutationValue = 1
  var _earlyStoppingScore = 0.95
  var _earlyStoppingFlag = true

  final val allowableStrategies = Seq("minimize", "maximize")
  final val allowableMutationStrategies = Seq("linear", "fixed")
  final val allowableMutationMagnitudeMode = Seq("random", "fixed")
  final val regressionMetrics: List[String] = List("rmse", "mse", "r2", "mae")
  final val classificationMetrics: List[String] = List("f1", "weightedPrecision", "weightedRecall", "accuracy")

  var _randomizer: scala.util.Random = scala.util.Random
  _randomizer.setSeed(_seed)

  def setLabelCol(value: String): this.type = {
    _labelCol = value
    this
  }

  def setFeaturesCol(value: String): this.type = {
    _featureCol = value
    this
  }

  def setTrainPortion(value: Double): this.type = {
    require(value < 1.0 & value > 0.0, "Training portion must be in the range > 0 and < 1")
    _trainPortion = value
    this
  }

  def setParallelism(value: Int): this.type = {
    //TODO: SET PARALLELISM VALIDATION CORRECTLY
    require(_parallelism < 10000, s"Parallelism above 10000 will result in cluster instability.")
    _parallelism = value
    this
  }

  def setKFold(value: Int): this.type = {
    _kFold = value
    _kFoldIteratorRange = Range(0, _kFold).par
    this
  }

  def setSeed(value: Long): this.type = {
    _seed = value
    this
  }

  def setOptimizationStrategy(value: String): this.type = {
    val valueLC = value.toLowerCase
    require(allowableStrategies.contains(valueLC),
      s"Optimization Strategy '$valueLC' is not a member of ${
        invalidateSelection(valueLC, allowableStrategies)
      }")
    _optimizationStrategy = valueLC
    this
  }

  def setFirstGenerationGenePool(value: Int): this.type = {
    require(value > 5, s"Values less than 5 for firstGenerationGenePool will require excessive generational mutation to converge")
    _firstGenerationGenePool = value
    this
  }

  def setNumberOfMutationGenerations(value: Int): this.type = {
    require(value > 0, s"Number of Generations must be greater than 0")
    _numberOfMutationGenerations = value
    this
  }

  def setNumberOfParentsToRetain(value: Int): this.type = {
    require(value > 0, s"Number of Parents must be greater than 0. '$value' is not a valid number.")
    _numberOfParentsToRetain = value
    this
  }

  def setNumberOfMutationsPerGeneration(value: Int): this.type = {
    require(value > 0, s"Number of Mutations per generation must be greater than 0. '$value' is not a valid number.")
    _numberOfMutationsPerGeneration = value
    this
  }

  def setGeneticMixing(value: Double): this.type = {
    require(value < 1.0 & value > 0.0,
      s"Mutation Aggressiveness must be in range (0,1). Current Setting of $value is not permitted.")
    _geneticMixing = value
    this
  }

  def setGenerationalMutationStrategy(value: String): this.type = {
    val valueLC = value.toLowerCase
    require(allowableMutationStrategies.contains(valueLC),
      s"Generational Mutation Strategy '$valueLC' is not a member of ${
        invalidateSelection(valueLC, allowableMutationStrategies)
      }")
    _generationalMutationStrategy = valueLC
    this
  }

  def setMutationMagnitudeMode(value: String): this.type = {
    val valueLC = value.toLowerCase
    require(allowableMutationMagnitudeMode.contains(valueLC),
      s"Mutation Magnitude Mode '$valueLC' is not a member of ${
        invalidateSelection(valueLC, allowableMutationMagnitudeMode)
      }")
    _mutationMagnitudeMode = valueLC
    this
  }

  def setFixedMutationValue(value: Int): this.type = {
    val maxMutationCount = modelConfigLength[RandomForestConfig]
    require(value <= maxMutationCount,
      s"Mutation count '$value' cannot exceed number of hyperparameters ($maxMutationCount)")
    require(value > 0, s"Mutation count '$value' must be greater than 0")
    _fixedMutationValue = value
    this
  }

  def setEarlyStoppingScore(value: Double): this.type = {
    _earlyStoppingScore = value
    this
  }

  def setEarlyStoppingFlag(value: Boolean): this.type = {
    _earlyStoppingFlag = value
    this
  }

  def getLabelCol: String = _labelCol

  def getFeaturesCol: String = _featureCol

  def getTrainPortion: Double = _trainPortion

  def getParallelism: Int = _parallelism

  def getKFold: Int = _kFold

  def getSeed: Long = _seed

  def getOptimizationStrategy: String = _optimizationStrategy

  def getFirstGenerationGenePool: Int = _firstGenerationGenePool

  def getNumberOfMutationGenerations: Int = _numberOfMutationGenerations

  def getNumberOfParentsToRetain: Int = _numberOfParentsToRetain

  def getNumberOfMutationsPerGeneration: Int = _numberOfMutationsPerGeneration

  def getGeneticMixing: Double = _geneticMixing

  def getGenerationalMutationStrategy: String = _generationalMutationStrategy

  def getMutationMagnitudeMode: String = _mutationMagnitudeMode

  def getFixedMutationValue: Int = _fixedMutationValue

  def getEarlyStoppingScore: Double = _earlyStoppingScore

  def getEarlyStoppingFlag: Boolean = _earlyStoppingFlag

  def totalModels: Int = (_numberOfMutationsPerGeneration * _numberOfMutationGenerations) + _firstGenerationGenePool

  def modelConfigLength[T: TypeTag]: Int = {
    typeOf[T].members.collect {
      case m: MethodSymbol if m.isCaseAccessor => m
    }.toList.length
  }

  def genTestTrain(data: DataFrame, seed: Long): Array[DataFrame] = {
    data.randomSplit(Array(_trainPortion, 1 - _trainPortion), seed)
  }

  def extractBoundaryDouble(param: String, boundaryMap: Map[String, (AnyVal, AnyVal)]): (Double, Double) = {
    val minimum = boundaryMap(param)._1.asInstanceOf[Double]
    val maximum = boundaryMap(param)._2.asInstanceOf[Double]
    (minimum, maximum)
  }

  def extractBoundaryInteger(param: String, boundaryMap: Map[String, (AnyVal, AnyVal)]): (Int, Int) = {
    val minimum = boundaryMap(param)._1.asInstanceOf[Double].toInt
    val maximum = boundaryMap(param)._2.asInstanceOf[Double].toInt
    (minimum, maximum)
  }

  def generateRandomDouble(param: String, boundaryMap: Map[String, (AnyVal, AnyVal)]): Double = {
    val (minimumValue, maximumValue) = extractBoundaryDouble(param, boundaryMap)
    minimumValue + _randomizer.nextDouble() * (maximumValue - minimumValue)
  }

  def generateRandomInteger(param: String, boundaryMap: Map[String, (AnyVal, AnyVal)]): Int = {
    val (minimumValue, maximumValue) = extractBoundaryInteger(param, boundaryMap)
    _randomizer.nextInt(maximumValue - minimumValue) + minimumValue
  }

  def generateRandomString(param: String, boundaryMap: Map[String, List[String]]): String = {
    _randomizer.shuffle(boundaryMap(param)).head
  }

  def coinFlip(): Boolean = {
    math.random < 0.5
  }

  def coinFlip(parent: Boolean, child: Boolean, p: Double): Boolean = {
    if (math.random < p) parent else child
  }

  def buildLayerArray(inputFeatureSize: Int, distinctClasses: Int, nLayers: Int,
                      hiddenLayerSizeAdjust: Int): Array[Int] = {

    val layerConstruct = new ArrayBuffer[Int]

    layerConstruct += inputFeatureSize

    (1 to nLayers).foreach { x =>
      layerConstruct += inputFeatureSize + nLayers - x + hiddenLayerSizeAdjust
    }
    layerConstruct += distinctClasses
    layerConstruct.result.toArray
  }

  def generateLayerArray(layerParam: String, layerSizeParam: String, boundaryMap: Map[String, (AnyVal, AnyVal)],
                         inputFeatureSize: Int, distinctClasses: Int): Array[Int] = {

    val layersToGenerate = generateRandomInteger(layerParam, boundaryMap)
    val hiddenLayerSizeAdjust = generateRandomInteger(layerSizeParam, boundaryMap)

    buildLayerArray(inputFeatureSize, distinctClasses, layersToGenerate, hiddenLayerSizeAdjust)

  }

  def getRandomIndeces(minimum: Int, maximum: Int, parameterCount: Int): List[Int] = {
    val fullIndexArray = List.range(0, maximum)
    val randomSeed = new scala.util.Random
    val count = minimum + randomSeed.nextInt((parameterCount - minimum) + 1)
    val adjCount = if (count < 1) 1 else count
    val shuffledArray = scala.util.Random.shuffle(fullIndexArray).take(adjCount)
    shuffledArray.sortWith(_ < _)
  }

  def getFixedIndeces(minimum: Int, maximum: Int, parameterCount: Int): List[Int] = {
    val fullIndexArray = List.range(0, maximum)
    val randomSeed = new scala.util.Random
    randomSeed.shuffle(fullIndexArray).take(parameterCount).sortWith(_ < _)
  }

  def generateMutationIndeces(minimum: Int, maximum: Int, parameterCount: Int,
                              mutationCount: Int): Array[List[Int]] = {
    val mutations = new ArrayBuffer[List[Int]]
    for (_ <- 0 to mutationCount) {
      _mutationMagnitudeMode match {
        case "random" => mutations += getRandomIndeces(minimum, maximum, parameterCount)
        case "fixed" => mutations += getFixedIndeces(minimum, maximum, parameterCount)
        case _ => new UnsupportedOperationException(
          s"Unsupported mutationMagnitudeMode ${_mutationMagnitudeMode}")
      }
    }
    mutations.result.toArray
  }

  def geneMixing(parent: Double, child: Double, parentMutationPercentage: Double): Double = {
    (parent * parentMutationPercentage) + (child * (1 - parentMutationPercentage))
  }

  def geneMixing(parent: Int, child: Int, parentMutationPercentage: Double): Int = {
    ((parent * parentMutationPercentage) + (child * (1 - parentMutationPercentage))).toInt
  }

  def geneMixing(parent: String, child: String): String = {
    val mixed = new ArrayBuffer[String]
    mixed += parent += child
    scala.util.Random.shuffle(mixed.toList).head
  }

  def geneMixing(parent: Array[Int], child: Array[Int], parentMutationPercentage: Double): Array[Int] = {

    val staticStart = parent.head
    val staticEnd = parent.last

    val parentHiddenLayers = parent.length - 2
    val childHiddenLayers = child.length - 2

    val parentMagnitude = parent(1) - staticStart
    val childMagnidue = child(1) - staticStart

    val hiddenLayerMix = geneMixing(parentHiddenLayers, childHiddenLayers, parentMutationPercentage)
    val sizeAdjustMix = geneMixing(parentMagnitude, childMagnidue, parentMutationPercentage)

    buildLayerArray(staticStart, staticEnd, hiddenLayerMix, sizeAdjustMix)

  }

  def calculateModelingFamilyRemainingTime(currentGen: Int, currentModel: Int): Double = {

    val modelsComplete = if (currentGen == 1) {
      currentModel
    } else {
      _firstGenerationGenePool + (_numberOfMutationsPerGeneration * (currentGen - 2) + currentModel)
    }

    (modelsComplete.toDouble / totalModels.toDouble) * 100

  }

}