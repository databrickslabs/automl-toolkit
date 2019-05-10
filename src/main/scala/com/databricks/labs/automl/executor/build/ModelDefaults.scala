package com.databricks.labs.automl.executor.build

object ModelDefaults {

  protected[build] def randomForestNumeric: Map[String, (Double, Double)] = Map(
    "numTrees" -> Tuple2(50.0, 1000.0),
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "subSamplingRate" -> Tuple2(0.5, 1.0)
  )

  protected[build] def randomForestString: Map[String, List[String]] = Map(
    "impurity" -> List("gini", "entropy"),
    "featureSubsetStrategy" -> List("auto")
  )

  protected[build] def treesNumeric: Map[String, (Double, Double)] = Map(
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "minInstancesPerNode" -> Tuple2(1.0, 50.0)
  )

  protected[build] def treesString: Map[String, List[String]] = Map(
    "impurity" -> List("gini", "entropy")
  )

  protected[build] def xgBoostNumeric: Map[String, (Double, Double)] = Map(
    "alpha" -> Tuple2(0.0, 1.0),
    "eta" -> Tuple2(0.1, 0.5),
    "gamma" -> Tuple2(0.0, 10.0),
    "lambda" -> Tuple2(0.1, 10.0),
    "maxDepth" -> Tuple2(3.0, 10.0),
    "subSample" -> Tuple2(0.4, 0.6),
    "minChildWeight" -> Tuple2(0.1, 10.0),
    "numRound" -> Tuple2(5.0, 25.0),
    "maxBins" -> Tuple2(25.0, 512.0),
    "trainTestRatio" -> Tuple2(0.2, 0.8)
  )

  protected[build] def mlpcNumeric: Map[String, (Double, Double)] = Map(
    "layers" -> Tuple2(1.0, 10.0),
    "maxIter" -> Tuple2(10.0, 100.0),
    "stepSize" -> Tuple2(0.01, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5),
    "hiddenLayerSizeAdjust" -> Tuple2(0.0, 50.0)
  )

  protected[build] def mlpcString: Map[String, List[String]] = Map(
    "solver" -> List("gd", "l-bfgs")
  )

  protected[build] def gbtNumeric: Map[String, (Double, Double)] = Map(
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxIter" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "minInstancesPerNode" -> Tuple2(1.0, 50.0),
    "stepSize" -> Tuple2(1E-4, 1.0)
  )

  protected[build] def gbtString: Map[String, List[String]] = Map(
    "impurity" -> List("gini", "entropy"),
    "lossType" -> List("logistic")
  )

  protected[build] def linearRegressionNumeric: Map[String, (Double, Double)] = Map(
    "elasticNetParams" -> Tuple2(0.0, 1.0),
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5)
  )

  protected[build] def linearRegressionString: Map[String, List[String]] = Map (
    "loss" -> List("squaredError", "huber")
  )

  protected[build] def logisticRegressionNumeric: Map[String, (Double, Double)] = Map(
    "elasticNetParams" -> Tuple2(0.0, 1.0),
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5)
  )

  protected[build] def svmNumeric: Map[String, (Double, Double)] = Map(
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5)
  )

}

