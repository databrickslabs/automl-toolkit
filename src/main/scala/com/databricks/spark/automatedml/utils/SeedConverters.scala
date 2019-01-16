package com.databricks.spark.automatedml.utils

import com.databricks.spark.automatedml.params._

import scala.collection.mutable.ListBuffer

trait SeedConverters {

  def generateRandomForestConfig(configMap: Map[String, Any]): RandomForestConfig = {
    RandomForestConfig(
      numTrees=configMap("numTrees").asInstanceOf[String].toInt,
      impurity=configMap("impurity").asInstanceOf[String],
      maxBins=configMap("maxBins").asInstanceOf[String].toInt,
      maxDepth=configMap("maxDepth").asInstanceOf[String].toInt,
      minInfoGain=configMap("minInfoGain").asInstanceOf[String].toDouble,
      subSamplingRate=configMap("subSamplingRate").asInstanceOf[String].toDouble,
      featureSubsetStrategy=configMap("featureSubsetStrategy").asInstanceOf[String]
    )
  }

  def generateMLPCConfig(configMap: Map[String, Any]): MLPCConfig = {

    var layers = ListBuffer[Int]()
    val stringLayers = configMap("layers").asInstanceOf[Array[String]]
    stringLayers.foreach{x => layers += x.toInt}

    MLPCConfig(
      layers=layers.result.toArray,
      maxIter=configMap("maxIter").asInstanceOf[String].toInt,
      solver=configMap("solver").asInstanceOf[String],
      stepSize=configMap("stepSize").asInstanceOf[String].toDouble,
      tol=configMap("tol").asInstanceOf[String].toDouble
    )
  }

  def generateTreesConfig(configMap: Map[String, Any]): TreesConfig = {
    TreesConfig(
      impurity=configMap("impurity").asInstanceOf[String],
      maxBins=configMap("maxBins").asInstanceOf[String].toInt,
      maxDepth=configMap("maxDepth").asInstanceOf[String].toInt,
      minInfoGain=configMap("minInfoGain").asInstanceOf[String].toDouble,
      minInstancesPerNode=configMap("minInstancesPerNode").asInstanceOf[String].toInt
    )
  }

  def generateGBTConfig(configMap: Map[String, Any]): GBTConfig = {
    GBTConfig(
      impurity=configMap("impurity").asInstanceOf[String],
      lossType=configMap("lossType").asInstanceOf[String],
      maxBins=configMap("maxBins").asInstanceOf[String].toInt,
      maxDepth=configMap("maxDepth").asInstanceOf[String].toInt,
      maxIter=configMap("maxIter").asInstanceOf[String].toInt,
      minInfoGain=configMap("minInfoGain").asInstanceOf[String].toDouble,
      minInstancesPerNode=configMap("minInstancesPerNode").asInstanceOf[String].toInt,
      stepSize=configMap("stepSize").asInstanceOf[String].toDouble
    )
  }

  def generateLogisticRegressionConfig(configMap: Map[String, Any]): LogisticRegressionConfig = {
    LogisticRegressionConfig(
      elasticNetParams = configMap("elasticNetParams").asInstanceOf[String].toDouble,
      fitIntercept = configMap("fitIntercept").asInstanceOf[String].toBoolean,
      maxIter = configMap("maxIter").asInstanceOf[String].toInt,
      regParam = configMap("regParam").asInstanceOf[String].toDouble,
      standardization = configMap("standardization").asInstanceOf[String].toBoolean,
      tolerance = configMap("tolerance").asInstanceOf[String].toDouble
    )
  }

  def generateLinearRegressionConfig(configMap: Map[String, Any]): LinearRegressionConfig = {
    LinearRegressionConfig(
      elasticNetParams = configMap("elasticNetParams").asInstanceOf[String].toDouble,
      fitIntercept = configMap("fitIntercept").asInstanceOf[String].toBoolean,
      loss = configMap("loss").asInstanceOf[String],
      maxIter = configMap("maxIter").asInstanceOf[String].toInt,
      regParam = configMap("regParam").asInstanceOf[String].toDouble,
      standardization = configMap("standardization").asInstanceOf[String].toBoolean,
      tolerance = configMap("tolerance").asInstanceOf[String].toDouble
    )
  }

  def generateSVMConfig(configMap: Map[String, Any]): SVMConfig = {
    SVMConfig(
      fitIntercept = configMap("fitIntercept").asInstanceOf[String].toBoolean,
      maxIter = configMap("maxIter").asInstanceOf[String].toInt,
      regParam = configMap("regParam").asInstanceOf[String].toDouble,
      standardization = configMap("standardization").asInstanceOf[String].toBoolean,
      tol = configMap("tol").asInstanceOf[String].toDouble
    )
  }

}
