package com.databricks.spark.automatedml.model.tools

import com.databricks.spark.automatedml.params.RandomForestConfig



trait ModelConfigGenerators extends SeedGenerator {

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
    }  yield RandomForestConfig(numTrees.toInt, impurity, maxBins.toInt, maxDepth.toInt, minInfoGain, subSamplingRate, featureSubsetStrategy)

  }

  protected[tools] def randomForestNumericArrayGenerator(config: RandomForestPermutationConfiguration):
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


  def randomForestPermutationGenerator(config: RandomForestPermutationConfiguration, countTarget: Int, seed: Long = 42L):
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

}
