package com.databricks.labs.automl.utils

import com.databricks.labs.automl.params._

import scala.collection.mutable.ListBuffer

trait SeedConverters {

  def generateXGBoostConfig(configMap: Map[String, Any]): XGBoostConfig = {
    XGBoostConfig(
      alpha = configMap("alpha").asInstanceOf[String].toDouble,
      eta = configMap("eta").asInstanceOf[String].toDouble,
      gamma = configMap("gamma").asInstanceOf[String].toDouble,
      lambda = configMap("lambda").asInstanceOf[String].toDouble,
      maxDepth = configMap("maxDepth").asInstanceOf[String].toInt,
      maxBins = configMap("maxBins").asInstanceOf[String].toInt,
      subSample = configMap("subSample").asInstanceOf[String].toDouble,
      minChildWeight = configMap("minChildWeight").asInstanceOf[String].toDouble,
      numRound = configMap("numRound").asInstanceOf[String].toInt,
      trainTestRatio = configMap("trainTestRatio").asInstanceOf[String].toDouble
    )
  }

  def generateRandomForestConfig(
    configMap: Map[String, Any]
  ): RandomForestConfig = {
    RandomForestConfig(
      numTrees = configMap("numTrees").asInstanceOf[String].toInt,
      impurity = configMap("impurity").asInstanceOf[String],
      maxBins = configMap("maxBins").asInstanceOf[String].toInt,
      maxDepth = configMap("maxDepth").asInstanceOf[String].toInt,
      minInfoGain = configMap("minInfoGain").asInstanceOf[String].toDouble,
      subSamplingRate =
        configMap("subSamplingRate").asInstanceOf[String].toDouble,
      featureSubsetStrategy =
        configMap("featureSubsetStrategy").asInstanceOf[String]
    )
  }

  def generateMLPCConfig(configMap: Map[String, Any]): MLPCConfig = {

    var layers = ListBuffer[Int]()
    val stringLayers = configMap("layers").asInstanceOf[Array[String]]
    stringLayers.foreach { x =>
      layers += x.toInt
    }

    MLPCConfig(
      layers = layers.result.toArray,
      maxIter = configMap("maxIter").asInstanceOf[String].toInt,
      solver = configMap("solver").asInstanceOf[String],
      stepSize = configMap("stepSize").asInstanceOf[String].toDouble,
      tolerance = configMap("tolerance").asInstanceOf[String].toDouble
    )
  }

  def generateTreesConfig(configMap: Map[String, Any]): TreesConfig = {
    TreesConfig(
      impurity = configMap("impurity").asInstanceOf[String],
      maxBins = configMap("maxBins").asInstanceOf[String].toInt,
      maxDepth = configMap("maxDepth").asInstanceOf[String].toInt,
      minInfoGain = configMap("minInfoGain").asInstanceOf[String].toDouble,
      minInstancesPerNode =
        configMap("minInstancesPerNode").asInstanceOf[String].toInt
    )
  }

  def generateGBTConfig(configMap: Map[String, Any]): GBTConfig = {
    GBTConfig(
      impurity = configMap("impurity").asInstanceOf[String],
      lossType = configMap("lossType").asInstanceOf[String],
      maxBins = configMap("maxBins").asInstanceOf[String].toInt,
      maxDepth = configMap("maxDepth").asInstanceOf[String].toInt,
      maxIter = configMap("maxIter").asInstanceOf[String].toInt,
      minInfoGain = configMap("minInfoGain").asInstanceOf[String].toDouble,
      minInstancesPerNode =
        configMap("minInstancesPerNode").asInstanceOf[String].toInt,
      stepSize = configMap("stepSize").asInstanceOf[String].toDouble
    )
  }

  def generateLogisticRegressionConfig(
    configMap: Map[String, Any]
  ): LogisticRegressionConfig = {
    LogisticRegressionConfig(
      elasticNetParams =
        configMap("elasticNetParams").asInstanceOf[String].toDouble,
      fitIntercept = configMap("fitIntercept").asInstanceOf[String].toBoolean,
      maxIter = configMap("maxIter").asInstanceOf[String].toInt,
      regParam = configMap("regParam").asInstanceOf[String].toDouble,
      standardization =
        configMap("standardization").asInstanceOf[String].toBoolean,
      tolerance = configMap("tolerance").asInstanceOf[String].toDouble
    )
  }

  def generateLinearRegressionConfig(
    configMap: Map[String, Any]
  ): LinearRegressionConfig = {
    LinearRegressionConfig(
      elasticNetParams =
        configMap("elasticNetParams").asInstanceOf[String].toDouble,
      fitIntercept = configMap("fitIntercept").asInstanceOf[String].toBoolean,
      loss = configMap("loss").asInstanceOf[String],
      maxIter = configMap("maxIter").asInstanceOf[String].toInt,
      regParam = configMap("regParam").asInstanceOf[String].toDouble,
      standardization =
        configMap("standardization").asInstanceOf[String].toBoolean,
      tolerance = configMap("tolerance").asInstanceOf[String].toDouble
    )
  }

  def generateSVMConfig(configMap: Map[String, Any]): SVMConfig = {
    SVMConfig(
      fitIntercept = configMap("fitIntercept").asInstanceOf[String].toBoolean,
      maxIter = configMap("maxIter").asInstanceOf[String].toInt,
      regParam = configMap("regParam").asInstanceOf[String].toDouble,
      standardization =
        configMap("standardization").asInstanceOf[String].toBoolean,
      tolerance = configMap("tolerance").asInstanceOf[String].toDouble
    )
  }

  def generateLightGBMConfig(configMap: Map[String, Any]): LightGBMConfig = {
    LightGBMConfig(
      baggingFraction =
        configMap("baggingFraction").asInstanceOf[String].toDouble,
      baggingFreq = configMap("baggingFreq").asInstanceOf[String].toInt,
      featureFraction =
        configMap("featureFraction").asInstanceOf[String].toDouble,
      learningRate = configMap("learningRate").asInstanceOf[String].toDouble,
      maxBin = configMap("maxBin").asInstanceOf[String].toInt,
      maxDepth = configMap("maxDepth").asInstanceOf[String].toInt,
      minSumHessianInLeaf =
        configMap("minSumHessianInLeaf").asInstanceOf[String].toDouble,
      numIterations = configMap("numIterations").asInstanceOf[String].toInt,
      numLeaves = configMap("numLeaves").asInstanceOf[String].toInt,
      boostFromAverage =
        configMap("boostFromAverage").asInstanceOf[String].toBoolean,
      lambdaL1 = configMap("lambdaL1").asInstanceOf[String].toDouble,
      lambdaL2 = configMap("lambdaL2").asInstanceOf[String].toDouble,
      alpha = configMap("alpha").asInstanceOf[String].toDouble,
      boostingType = configMap("boostingType").asInstanceOf[String]
    )
  }

}
