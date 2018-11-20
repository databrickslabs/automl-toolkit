package com.databricks.spark.automatedml.params

trait Defaults {

  final val _supportedModels: Array[String] = Array(
    "GBT",
    "RandomForest",
    "LinearRegression",
    "LogisticRegression",
    "MLPC",
    "SVM"
  )

  val _geneticTunerDefaults = GeneticConfig(
    kFold = 5,
    trainPortion = 3,
    seed = 42L,
    firstGenerationGenePool = 20,
    numberOfGenerations = 10,
    numberOfParentsToRetain = 3,
    numberOfMutationsPerGeneration = 10,
    geneticMixing = 0.7,
    generationalMutationStrategy = "linear",
    fixedMutationValue = 1,
    mutationMagnitudeMode = "fixed"
  )

  val _fillConfigDefaults = FillConfig(
    numericFillStat = "mean",
    characterFillStat = "max",
    modelSelectionDistinctThreshold = 10
  )

  val _outlierConfigDefaults = OutlierConfig(
    filterBounds = "both",
    lowerFilterNTile = 0.02,
    upperFilterNTile = 0.98,
    filterPrecision = 0.01,
    continuousDataThreshold = 50,
    fieldsToIgnore = Array("")
  )

  val _pearsonConfigDefaults = PearsonConfig(
    filterStatistic = "pearsonStat",
    filterDirection = "greater",
    filterManualValue = 0.0,
    filterMode = "auto",
    autoFilterNTile = 0.75
  )

  val _covarianceConfigDefaults = CovarianceConfig(
    correlationCutoffLow = -0.8,
    correlationCutoffHigh = 0.8
  )

  val _rfDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "numTrees" -> Tuple2(50.0, 1000.0),
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "subSamplingRate" -> Tuple2(0.5, 1.0)
  )

  val _rfDefaultStringBoundaries = Map(
    "impurity" -> List("gini", "entropy"),
    "featureSubsetStrategy" -> List("all", "sqrt", "log2", "onethird")
  )

  val _mlpcDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "layers" -> Tuple2(1.0, 10.0),
    "maxIter" -> Tuple2(100.0, 10000.0),
    "stepSize" -> Tuple2(0.01, 1.0),
    "tol" -> Tuple2(1E-9, 1E-5),
    "hiddenLayerSizeAdjust" -> Tuple2(0.0, 50.0)
  )

  val _mlpcDefaultStringBoundaries: Map[String, List[String]] = Map(
    "solver" -> List("gd", "l-bfgs")
  )

  val _gbtDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "minInstancesPerNode" -> Tuple2(1.0, 50.0),
    "stepSize" -> Tuple2(0.0, 1.0)
  )

  val _gbtDefaultStringBoundaries: Map[String, List[String]] = Map(
    "impurity" -> List("gini", "entropy"),
    "lossType" -> List("logistic")
  )

  val _scoringDefaultClassifier = "f1"
  val _scoringOptimizationStrategyClassifier = "maximize"
  val _scoringDefaultRegressor = "rmse"
  val _scoringOptimizationStrategyRegressor = "minimize"


}