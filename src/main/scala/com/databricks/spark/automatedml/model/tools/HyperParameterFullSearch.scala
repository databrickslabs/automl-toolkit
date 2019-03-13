package com.databricks.spark.automatedml.model.tools

import com.databricks.spark.automatedml.params.{Defaults, RandomForestConfig}

import scala.collection.mutable.ArrayBuffer
import util.Random


class HyperParameterFullSearch extends SeedGenerator with Defaults {

  var _modelFamily = ""
  var _modelType = ""
  var _permutationCount = 10
  var _indexMixingMode = "linear"

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

  def getModelFamily: String = _modelFamily
  def getModelType: String = _modelType
  def getPermutationCount: Int = _permutationCount
  def getIndexMixingMode: String = _indexMixingMode


  private def randomIndexSelection(numericArrays: Array[Array[Double]]): NumericArrayCollection = {

    val bufferContainer = new ArrayBuffer[Array[Double]]()

    numericArrays.foreach{ x =>
      bufferContainer += Random.shuffle(x.toList).toArray
    }

    val arrayRandomHolder = bufferContainer.result.toArray

    val randomlySelectedPayload = arrayRandomHolder.map(x => x(0))

    val remainingArrays = arrayRandomHolder.map(x => x.drop(1))

    NumericArrayCollection(randomlySelectedPayload, remainingArrays)

  }

  private def staticIndexSelection(numericArrays: Array[Array[Double]]): NumericArrayCollection = {

    val selectedPayload = numericArrays.map(x => x(0))

    val remainingArrays = numericArrays.map(x => x.drop(1))

    NumericArrayCollection(selectedPayload, remainingArrays)

  }

  private def extractContinuousBoundaries(parameter: Tuple2[Double, Double]): NumericBoundaries = {
    NumericBoundaries(parameter._1, parameter._2)
  }

  private def selectStringIndex(availableParams: List[String], currentIterator: Int): StringSelectionReturn = {
    val listLength = availableParams.length
    val idxSelection = if(currentIterator >= listLength) 0 else currentIterator

    StringSelectionReturn(availableParams(idxSelection), idxSelection)
  }

  def generateRandomForestHyperParameters(numericBoundaries: Map[String, (Double, Double)],
                                                  stringBoundaries: Map[String, List[String]]): Array[RandomForestConfig] = {

    var outputPayload = new ArrayBuffer[RandomForestConfig]()

    // figure out the number of permutations to generate
    val numericValuesCollection = stringBoundaries.size
    val seedsToGenerate = _permutationCount / numericValuesCollection

    /**
      * General Guidelines:
      *
      * - For String / Boolean values: Re-use as an iterator to continue to select index positions through the loop.
      * - For Continuous Variables:
      * -- mode: "Linear" - Generate uniformly sized Arrays, then build by index position to create the config.
      * -- mode: "Random" - Generate uniformly sized Array, build by random combination without replacement.
      */


    val impurityValues = _modelType match {
      case "regressor" => List("variance")
      case _ => stringBoundaries("impurity")
    }

    //TODO: generate the numeric Arrays for each of the hyper parameters


    val numTreesArray = generateLinearIntSpace(
      extractContinuousBoundaries(numericBoundaries("numTrees")), seedsToGenerate)
    val maxBinsArray = generateLinearIntSpace(
      extractContinuousBoundaries(numericBoundaries("maxBins")), seedsToGenerate)
    val maxDepthArray = generateLinearIntSpace(
      extractContinuousBoundaries(numericBoundaries("maxDepth")), seedsToGenerate)
    val minInfoGainArray = generateLogSpace(
      extractContinuousBoundaries(numericBoundaries("minInfoGain")), seedsToGenerate)
    val subSamplingRateArray = generateLinearIntSpace(
      extractContinuousBoundaries(numericBoundaries("subSamplingRate")), seedsToGenerate
    )

    // Create some index values
    var _impurityIdx = 0
    var _featureSubsetStrategyIdx = 0

    //TODO: within the loop, call the index selection, then add in the parameters for each string value and build
    // the RandomForestConfig collection.

    var numericArrays = Array(numTreesArray, maxBinsArray, maxDepthArray, minInfoGainArray, subSamplingRateArray)

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

case class NumericBoundaries(
                              minimum: Double,
                              maximum: Double
                            )

case class NumericArrayCollection(
                                  selectedPayload: Array[Double],
                                  remainingPayload: Array[Array[Double]]
                                 )
case class StringSelectionReturn(
                                 selectedStringValue: String,
                                 IndexCounterStatus: Int
                                )