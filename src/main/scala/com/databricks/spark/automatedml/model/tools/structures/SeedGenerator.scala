package com.databricks.spark.automatedml.model.tools.structures

import scala.collection.mutable.ArrayBuffer
import scala.math._
import scala.reflect.ClassTag
import scala.util.Random

trait SeedGenerator {


  def generateLinearIntSpace(boundaries: NumericBoundaries, generatorCount: Int): Array[Double] = {

    val integerSpace = new ArrayBuffer[Double]()

    val generatedDoubles = generateLinearSpace(boundaries, generatorCount)

    generatedDoubles.foreach{ x => integerSpace += x.round}

    integerSpace.result.toArray

  }

  def generateLinearSpace(boundaries: NumericBoundaries, generatorCount: Int): Array[Double] = {

    val space = new ArrayBuffer[Double]

    val iteratorDelta = (boundaries.maximum - boundaries.minimum) / (generatorCount.toDouble - 1.0)

    for(i <- 0 until generatorCount - 1) {
      space += boundaries.minimum + i * iteratorDelta
    }
    space += boundaries.maximum
    space.result.toArray
  }

  def convertToLog(minScale: Double, maxScale: Double, value: Double): Double = {

    val minVal = if(minScale == 0.0) 1.0E-10 else minScale

    val b = log(maxScale / minVal) / (maxScale - minVal)

    val a = maxScale / exp(b * maxScale)

    a * exp(b * value)
  }

  def generateLogSpace(boundaries: NumericBoundaries, generatorCount: Int): Array[Double] = {

    val space = new ArrayBuffer[Double]

    val linearSpace = generateLinearSpace(boundaries, generatorCount)

    linearSpace.foreach{ x=>
      space += convertToLog(boundaries.minimum, boundaries.maximum, x)
    }

    space.result.toArray

  }

  private[SeedGenerator] def getNthRoot(n: Double, root: Double): Double = {
    pow(exp(1.0/root), log(n))
  }

  def getNumberOfElements(numericBoundaries: Map[String, (Double, Double)]): Int = {
    numericBoundaries.keys.size
  }

  def getPermutationCounts(targetIterations: Int, numberOfElements: Int): Int = {

    getNthRoot(targetIterations.toDouble, numberOfElements.toDouble).ceil.toInt

  }

  protected[tools] def randomSampleArray[T: ClassTag](hyperParameterArray: Array[T], sampleCount: Int, seed:Long=42L): Array[T] = {

    val randomSeed = new Random(seed)
    Array.fill(sampleCount)(hyperParameterArray(randomSeed.nextInt(hyperParameterArray.length)))

  }

  protected[tools] def extractContinuousBoundaries(parameter: Tuple2[Double, Double]): NumericBoundaries = {
    NumericBoundaries(parameter._1, parameter._2)
  }

  protected[tools] def selectStringIndex(availableParams: List[String], currentIterator: Int): StringSelectionReturn = {
    val listLength = availableParams.length
    val idxSelection = if(currentIterator >= listLength) 0 else currentIterator

    StringSelectionReturn(availableParams(idxSelection), idxSelection)
  }

  protected[tools] def selectCoinFlip(currentIterator: Int): Boolean = {
    if (currentIterator.toDouble % 2.0 == 0.0) true else false
  }

  protected[tools] def staticIndexSelection(numericArrays: Array[Array[Double]]): NumericArrayCollection = {

    val selectedPayload = numericArrays.map(x => x(0))

    val remainingArrays = numericArrays.map(x => x.drop(1))

    NumericArrayCollection(selectedPayload, remainingArrays)

  }

  protected[tools] def randomIndexSelection(numericArrays: Array[Array[Double]]): NumericArrayCollection = {

    val bufferContainer = new ArrayBuffer[Array[Double]]()

    numericArrays.foreach{ x =>
      bufferContainer += Random.shuffle(x.toList).toArray
    }

    val arrayRandomHolder = bufferContainer.result.toArray

    val randomlySelectedPayload = arrayRandomHolder.map(x => x(0))

    val remainingArrays = arrayRandomHolder.map(x => x.drop(1))

    NumericArrayCollection(randomlySelectedPayload, remainingArrays)

  }

  /**
    * Calculates the number of possible additional permutations to be added to the search space for string values
    * @param stringBoundaries The string boundary payload for a modeling family
    * @return Int representing any additional permutations on the numeric body that will need to be generated in order
    *         to attempt to reach the target unique hyperparameter search space
    */
  protected[tools] def stringBoundaryPermutationCalculator(stringBoundaries: Map[String, List[String]]): Int = {

    var uniqueValues = 0

    stringBoundaries.foreach{ x =>
      uniqueValues += x._2.length - 1
    }

    uniqueValues
  }

}
