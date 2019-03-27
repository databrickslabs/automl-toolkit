package com.databricks.spark.automatedml.model.tools.structures

import com.databricks.spark.automatedml.params.RandomForestConfig

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe._

trait ModelConfigGenerators extends SeedGenerator {

  def getCaseClassNames[T: TypeTag]: List[String] = typeOf[T].members.sorted.collect {
    case m: MethodSymbol if m.isCaseAccessor => m.name.toString
  }

  def randomForestConfigGenerator(randomForestPermutationCollection: RandomForestPermutationCollection):
  Array[RandomForestConfig] = {

    for {
      numTrees <- randomForestPermutationCollection.numTreesArray
      impurity <- randomForestPermutationCollection.impurityArray
      maxBins <- randomForestPermutationCollection.maxBinsArray
      maxDepth <- randomForestPermutationCollection.maxDepthArray
      minInfoGain <- randomForestPermutationCollection.minInfoGainArray
      subSamplingRate <- randomForestPermutationCollection.subSamplingRateArray
      featureSubsetStrategy <- randomForestPermutationCollection.featureSubsetStrategyArray
    } yield RandomForestConfig(numTrees.toInt, impurity, maxBins.toInt, maxDepth.toInt, minInfoGain, subSamplingRate, featureSubsetStrategy)

  }

  protected[tools] def randomForestNumericArrayGenerator(config: PermutationConfiguration):
  RandomForestNumericArrays = {

    RandomForestNumericArrays(
      numTreesArray = generateLinearIntSpace(extractContinuousBoundaries(config.numericBoundaries("numTrees")),
        config.permutationTarget),
      maxBinsArray = generateLinearIntSpace(
        extractContinuousBoundaries(config.numericBoundaries("maxBins")), config.permutationTarget),
      maxDepthArray = generateLinearIntSpace(
        extractContinuousBoundaries(config.numericBoundaries("maxDepth")), config.permutationTarget),
      minInfoGainArray = generateLogSpace(
        extractContinuousBoundaries(config.numericBoundaries("minInfoGain")), config.permutationTarget),
      subSamplingRateArray = generateLinearSpace(
        extractContinuousBoundaries(config.numericBoundaries("subSamplingRate")), config.permutationTarget)
    )

  }

  def randomForestPermutationGenerator(config: PermutationConfiguration, countTarget: Int, seed: Long = 42L):
  Array[RandomForestConfig] = {

    // Get the number of permutations to generate
    val numericPayloads = randomForestNumericArrayGenerator(config)

    val fullPermutationConfig = RandomForestPermutationCollection(
      numTreesArray = numericPayloads.numTreesArray,
      maxBinsArray = numericPayloads.maxBinsArray,
      maxDepthArray = numericPayloads.maxDepthArray,
      minInfoGainArray = numericPayloads.minInfoGainArray,
      subSamplingRateArray = numericPayloads.subSamplingRateArray,
      impurityArray = config.stringBoundaries("impurity").toArray,
      featureSubsetStrategyArray = config.stringBoundaries("featureSubsetStrategy").toArray
    )

    val permutationCollection = randomForestConfigGenerator(fullPermutationConfig)

    randomSampleArray(permutationCollection, countTarget, seed)

  }

  def convertRandomForestResultToConfig(predictionDataFrame: DataFrame): Array[RandomForestConfig] = {

    val collectionBuffer = new ArrayBuffer[RandomForestConfig]()

    val dataCollection = predictionDataFrame
      .select(getCaseClassNames[RandomForestConfig] map col :_*)
      .collect()

    dataCollection.foreach{ x =>

      collectionBuffer += RandomForestConfig(
        numTrees = x(0).toString.toInt,
        impurity = x(1).toString,
        maxBins = x(2).toString.toInt,
        maxDepth = x(3).toString.toInt,
        minInfoGain = x(4).toString.toDouble,
        subSamplingRate = x(5).toString.toDouble,
        featureSubsetStrategy = x(6).toString
      )

    }

    collectionBuffer.result.toArray
  }

}
