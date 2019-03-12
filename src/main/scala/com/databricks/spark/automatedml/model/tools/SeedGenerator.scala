package com.databricks.spark.automatedml.model.tools

import scala.collection.mutable.ArrayBuffer
import scala.math._

trait SeedGenerator {


  def generateLinearIntSpace(minimum: Int, maximum: Int, generatorCount: Int): Array[Int] = {

    val integerSpace = new ArrayBuffer[Int]()

    val generatedDoubles = generateLinearSpace(minimum.toDouble, maximum.toDouble, generatorCount)

    generatedDoubles.foreach{ x => integerSpace += x.round.toInt}

    integerSpace.result.toArray

  }

  def generateLinearSpace(minimum: Double, maximum: Double, generatorCount: Int): Array[Double] = {

    val space = new ArrayBuffer[Double]

    val iteratorDelta = (maximum - minimum) / (generatorCount.toDouble - 1.0)

    for(i <- 0 until generatorCount - 1) {
      space += minimum + i * iteratorDelta
    }
    space += maximum
    space.result.toArray
  }

  def convertToLog(minScale: Double, maxScale: Double, value: Double): Double = {

    val b = log(maxScale / minScale) / (maxScale - minScale)

    val a = maxScale / exp(b * maxScale)

    a * exp(b * value)
  }

  def generateLogSpace(minimum: Double, maximum: Double, generatorCount: Int): Array[Double] = {

    val space = new ArrayBuffer[Double]

    val linearSpace = generateLinearSpace(minimum, maximum, generatorCount)

    linearSpace.foreach{ x=>
      space += convertToLog(minimum, maximum, x)
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