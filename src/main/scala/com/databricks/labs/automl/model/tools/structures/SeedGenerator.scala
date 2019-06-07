package com.databricks.labs.automl.model.tools.structures

import com.databricks.labs.automl.params.MLPCConfig

import scala.collection.mutable.ArrayBuffer
import scala.math._
import scala.reflect.ClassTag
import scala.util.Random

trait SeedGenerator {

  def generateLinearIntSpace(boundaries: NumericBoundaries,
                             generatorCount: Int): Array[Double] = {

    val integerSpace = new ArrayBuffer[Double]()

    val generatedDoubles = generateLinearSpace(boundaries, generatorCount)

    generatedDoubles.foreach { x =>
      integerSpace += x.round
    }

    integerSpace.result.toArray

  }

  def generateLinearSpace(boundaries: NumericBoundaries,
                          generatorCount: Int): Array[Double] = {

    val space = new ArrayBuffer[Double]

    val iteratorDelta = (boundaries.maximum - boundaries.minimum) / (generatorCount.toDouble - 1.0)

    for (i <- 0 until generatorCount - 1) {
      space += boundaries.minimum + i * iteratorDelta
    }
    space += boundaries.maximum
    space.result.toArray
  }

  def convertToLog(minScale: Double,
                   maxScale: Double,
                   value: Double): Double = {

    val minVal = if (minScale == 0.0) 1.0E-10 else minScale

    val b = log(maxScale / minVal) / (maxScale - minVal)

    val a = maxScale / exp(b * maxScale)

    a * exp(b * value)
  }

  def generateLogSpace(boundaries: NumericBoundaries,
                       generatorCount: Int): Array[Double] = {

    val space = new ArrayBuffer[Double]

    val linearSpace = generateLinearSpace(boundaries, generatorCount)

    linearSpace.foreach { x =>
      space += convertToLog(boundaries.minimum, boundaries.maximum, x)
    }

    space.result.toArray

  }

  private[tools] def constructLayerArray(inputFeatureSize: Int,
                                         distinctClasses: Int,
                                         layerCount: Int,
                                         sizeAdjustment: Int): Array[Int] = {

    val layerConstruct = new ArrayBuffer[Int]

    layerConstruct += inputFeatureSize

    (1 to layerCount).foreach(
      x => layerConstruct += inputFeatureSize + layerCount - x + sizeAdjustment
    )
    layerConstruct += distinctClasses
    layerConstruct.result.toArray
  }

  def generateArraySpace(layerBoundaryLow: Int,
                         layerBoundaryHigh: Int,
                         hiddenBoundaryLow: Int,
                         hiddenBoundaryHigh: Int,
                         inputFeatureSize: Int,
                         distinctClasses: Int,
                         generatorCount: Int): Array[Array[Int]] = {

    val outputBuffer = new ArrayBuffer[Array[Int]]()

    // Generate the layer Boundary space and the size adjustment space
    val layerBoundaries = generateLinearIntSpace(
      NumericBoundaries(layerBoundaryLow, layerBoundaryHigh),
      generatorCount
    )
    val sizeAdjustmentBoundaries = generateLinearIntSpace(
      NumericBoundaries(hiddenBoundaryLow, hiddenBoundaryHigh),
      generatorCount
    )

    (0 until generatorCount).foreach { x =>
      outputBuffer += constructLayerArray(
        inputFeatureSize,
        distinctClasses,
        layerBoundaries(x).toInt,
        sizeAdjustmentBoundaries(x).toInt
      )
    }
    outputBuffer.result.toArray
  }

  private[SeedGenerator] def getNthRoot(n: Double, root: Double): Double = {
    pow(exp(1.0 / root), log(n))
  }

  def getNumberOfElements(
    numericBoundaries: Map[String, (Double, Double)]
  ): Int = {
    numericBoundaries.keys.size
  }

  def getPermutationCounts(targetIterations: Int,
                           numberOfElements: Int): Int = {

    getNthRoot(targetIterations.toDouble, numberOfElements.toDouble).ceil.toInt

  }

  protected[tools] def randomSampleArray[T: ClassTag](
    hyperParameterArray: Array[T],
    sampleCount: Int,
    seed: Long = 42L
  ): Array[T] = {

    val randomSeed = new Random(seed)
    Array.fill(sampleCount)(
      hyperParameterArray(randomSeed.nextInt(hyperParameterArray.length))
    )

  }

  protected[tools] def extractContinuousBoundaries(
    parameter: Tuple2[Double, Double]
  ): NumericBoundaries = {
    NumericBoundaries(parameter._1, parameter._2)
  }

  protected[tools] def selectStringIndex(
    availableParams: List[String],
    currentIterator: Int
  ): StringSelectionReturn = {
    val listLength = availableParams.length
    val idxSelection = if (currentIterator >= listLength) 0 else currentIterator

    StringSelectionReturn(availableParams(idxSelection), idxSelection)
  }

  protected[tools] def selectCoinFlip(currentIterator: Int): Boolean = {
    if (currentIterator.toDouble % 2.0 == 0.0) true else false
  }

  protected[tools] def staticIndexSelection(
    numericArrays: Array[Array[Double]]
  ): NumericArrayCollection = {

    val selectedPayload = numericArrays.map(x => x(0))

    val remainingArrays = numericArrays.map(x => x.drop(1))

    NumericArrayCollection(selectedPayload, remainingArrays)

  }

  protected[tools] def randomIndexSelection(
    numericArrays: Array[Array[Double]]
  ): NumericArrayCollection = {

    val bufferContainer = new ArrayBuffer[Array[Double]]()

    numericArrays.foreach { x =>
      bufferContainer += Random.shuffle(x.toList).toArray
    }

    val arrayRandomHolder = bufferContainer.result.toArray

    val randomlySelectedPayload = arrayRandomHolder.map(x => x(0))

    val remainingArrays = arrayRandomHolder.map(x => x.drop(1))

    NumericArrayCollection(randomlySelectedPayload, remainingArrays)

  }

  protected[tools] def mlpcStaticIndexSelection(
    numericArrays: MLPCNumericArrays
  ): MLPCArrayCollection = {

    val layersArray = numericArrays.layersArray
    val selectedLayers = layersArray.take(1)(0)
    val updatedLayersArray = layersArray.drop(1)

    val maxIterArray = numericArrays.maxIterArray
    val selectedMaxIter = maxIterArray.take(1)(0)
    val updatedMaxIterArray = maxIterArray.drop(1)

    val stepSizeArray = numericArrays.stepSizeArray
    val selectedStepSize = stepSizeArray.take(1)(0)
    val updatedStepSizeArray = stepSizeArray.drop(1)

    val tolArray = numericArrays.toleranceArray
    val selectedTol = tolArray.take(1)(0)
    val updateTolArray = tolArray.drop(1)

    val remainingArrays = MLPCNumericArrays(
      layersArray = updatedLayersArray,
      maxIterArray = updatedMaxIterArray,
      stepSizeArray = updatedStepSizeArray,
      toleranceArray = updateTolArray
    )

    MLPCArrayCollection(
      MLPCConfig(
        layers = selectedLayers,
        maxIter = selectedMaxIter.toInt,
        solver = "placeholder",
        stepSize = selectedStepSize,
        tolerance = selectedTol
      ),
      remainingArrays
    )

  }

  protected[tools] def mlpcLayersExtractor(layers: Array[Int]): (Int, Int) = {

    val hiddenLayersSizeAdjust =
      if (layers.length > 2) layers(1) - layers(0) else 0
    val layerCount = layers.length - 2

    (layerCount, hiddenLayersSizeAdjust)
  }

  protected[tools] def mlpcRandomIndexSelection(
    numericArrays: MLPCNumericArrays
  ): MLPCArrayCollection = {

    val shuffledData = MLPCNumericArrays(
      layersArray = Random.shuffle(numericArrays.layersArray.toList).toArray,
      maxIterArray = Random.shuffle(numericArrays.maxIterArray.toList).toArray,
      stepSizeArray = Random.shuffle(numericArrays.stepSizeArray.toList).toArray,
      toleranceArray =
        Random.shuffle(numericArrays.toleranceArray.toList).toArray
    )

    mlpcStaticIndexSelection(shuffledData)
  }

  /**
    * Calculates the number of possible additional permutations to be added to the search space for string values
    * @param stringBoundaries The string boundary payload for a modeling family
    * @return Int representing any additional permutations on the numeric body that will need to be generated in order
    *         to attempt to reach the target unique hyperparameter search space
    */
  protected[tools] def stringBoundaryPermutationCalculator(
    stringBoundaries: Map[String, List[String]]
  ): Int = {

    var uniqueValues = 0

    stringBoundaries.foreach { x =>
      uniqueValues += x._2.length - 1
    }

    uniqueValues
  }

}
