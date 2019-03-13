package com.databricks.spark.automatedml.model.tools

import scala.collection.mutable.ArrayBuffer
import scala.math._

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

    val b = log(maxScale / minScale) / (maxScale - minScale)

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

}

/**
case class RandomForestConfig(
                               numTrees: Int,
                               impurity: String,
                               maxBins: Int,
                               maxDepth: Int,
                               minInfoGain: Double,
                               subSamplingRate: Double,
                               featureSubsetStrategy: String
                             )
  */