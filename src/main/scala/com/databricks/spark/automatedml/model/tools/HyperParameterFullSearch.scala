package com.databricks.spark.automatedml.model.tools

import com.databricks.spark.automatedml.model.tools.structures.{ModelConfigGenerators, PermutationConfiguration}
import com.databricks.spark.automatedml.params.{Defaults, RandomForestConfig}

import scala.collection.mutable.ArrayBuffer


class HyperParameterFullSearch extends Defaults with ModelConfigGenerators {

  var _modelFamily = ""
  var _modelType = ""
  var _permutationCount = 10
  var _indexMixingMode = "linear"
  var _arraySeed = 42L

  private val allowableMixingModes = List("linear", "random")

  def setModelFamily(value: String): this.type = {
    require(_supportedModels.contains(value), s"${this.getClass.toString} error! Model Family $value is not supported." +
      s"\n\t Supported families: ${_supportedModels.mkString(", ")}")
    _modelFamily = value
    this
  }

  def setModelType(value: String): this.type = {
    value match {
      case "classifier" => _modelType = value
      case "regressor" => _modelType = value
      case _ => throw new UnsupportedOperationException(s"Model type $value is not supported.")
    }
    this
  }

  def setPermutationCount(value: Int): this.type = {
    _permutationCount = value
    this
  }

  def setIndexMixingMode(value: String): this.type = {
    require(allowableMixingModes.contains(value), s"Index Mixing mode $value is not supported.  Allowable modes are: " +
      s"${allowableMixingModes.mkString(", ")}")
    _indexMixingMode = value
    this
  }

  def setArraySeed(value: Long): this.type = {
    _arraySeed = value
    this
  }

  def getModelFamily: String = _modelFamily
  def getModelType: String = _modelType
  def getPermutationCount: Int = _permutationCount
  def getIndexMixingMode: String = _indexMixingMode
  def getArraySeed: Long = _arraySeed

  def initialGenerationSeedRandomForest(numericBoundaries: Map[String, (Double, Double)],
                            stringBoundaries: Map[String, List[String]]): Array[RandomForestConfig] = {

    var outputPayload = new ArrayBuffer[RandomForestConfig]()

    val impurityValues = _modelType match {
      case "regressor" => List("variance")
      case _ => stringBoundaries("impurity")
    }

    // Set the config object
    val rfConfig = PermutationConfiguration(
      permutationTarget = _permutationCount,
      numericBoundaries = numericBoundaries,
      stringBoundaries = stringBoundaries
    )

    // Generate the permutation collections

    val generatedArrays = randomForestNumericArrayGenerator(rfConfig)

    // Create some index values
    var _impurityIdx = 0
    var _featureSubsetStrategyIdx = 0

    var numericArrays = Array(generatedArrays.numTreesArray, generatedArrays.maxBinsArray,
      generatedArrays.maxDepthArray, generatedArrays.minInfoGainArray, generatedArrays.subSamplingRateArray)

    // Main builder loop
    for (i <- 1 to _permutationCount) {

      val selectedIndeces = _indexMixingMode match {
        case "linear" => staticIndexSelection(numericArrays)
        case "random" => randomIndexSelection(numericArrays)
        case _ => throw new UnsupportedOperationException(s"index mixing mode ${_indexMixingMode} is not supported.")
      }

      numericArrays = selectedIndeces.remainingPayload

      // Handle the string value selections
      val impurityLoop = selectStringIndex(impurityValues, _impurityIdx)

      _impurityIdx = impurityLoop.IndexCounterStatus

      val featureSubsetStrategyLoop = selectStringIndex(stringBoundaries("featureSubsetStrategy"),
        _featureSubsetStrategyIdx)

      _featureSubsetStrategyIdx = featureSubsetStrategyLoop.IndexCounterStatus

      outputPayload += RandomForestConfig(
        numTrees = selectedIndeces.selectedPayload(0).toInt,
        impurity = impurityLoop.selectedStringValue,
        maxBins = selectedIndeces.selectedPayload(1).toInt,
        maxDepth = selectedIndeces.selectedPayload(2).toInt,
        minInfoGain = selectedIndeces.selectedPayload(3),
        subSamplingRate = selectedIndeces.selectedPayload(4),
        featureSubsetStrategy = featureSubsetStrategyLoop.selectedStringValue
      )
      _impurityIdx += 1
      _featureSubsetStrategyIdx += 1
    }

  outputPayload.result.toArray

  }



}

