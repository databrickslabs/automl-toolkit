package com.databricks.spark.automatedml.params

trait Defaults {

  final val _supportedModels: Array[String] = Array(
    "GBT",
    "Trees",
    "RandomForest",
    "LinearRegression",
    "LogisticRegression",
    "MLPC",
    "SVM"
  )

  def _defaultModelingFamily: String = "RandomForest"

  def _defaultLabelCol: String = "label"

  def _defaultFeaturesCol: String = "features"

  def _defaultNAFillFlag: Boolean = true

  def _defaultVarianceFilterFlag: Boolean = true

  def _defaultOutlierFilterFlag: Boolean = true

  def _defaultPearsonFilterFlag: Boolean = true

  def _defaultCovarianceFilterFlag: Boolean = true

  def _defaultScalingFlag: Boolean = false

  def _defaultDateTimeConversionType: String = "split"

  def _defaultFieldsToIgnoreInVector: Array[String] = Array.empty[String]

  def _geneticTunerDefaults = GeneticConfig(
    parallelism = 20,
    kFold = 5,
    trainPortion = 0.8,
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

  def _fillConfigDefaults = FillConfig(
    numericFillStat = "mean",
    characterFillStat = "max",
    modelSelectionDistinctThreshold = 10
  )

  def _outlierConfigDefaults = OutlierConfig(
    filterBounds = "both",
    lowerFilterNTile = 0.02,
    upperFilterNTile = 0.98,
    filterPrecision = 0.01,
    continuousDataThreshold = 50,
    fieldsToIgnore = Array("")
  )

  def _pearsonConfigDefaults = PearsonConfig(
    filterStatistic = "pearsonStat",
    filterDirection = "greater",
    filterManualValue = 0.0,
    filterMode = "auto",
    autoFilterNTile = 0.75
  )

  def _covarianceConfigDefaults = CovarianceConfig(
    correlationCutoffLow = -0.8,
    correlationCutoffHigh = 0.8
  )

  def _scalingConfigDefaults = ScalingConfig(
    scalerType = "minMax",
    scalerMin = 0.0,
    scalerMax = 1.0,
    standardScalerMeanFlag = false,
    standardScalerStdDevFlag = true,
    pNorm = 2.0
  )

  def _dataPrepConfigDefaults = DataPrepConfig(
    naFillFlag = true,
    varianceFilterFlag = true,
    outlierFilterFlag = false,
    pearsonFilterFlag = true,
    covarianceFilterFlag = true,
    scalingFlag = false
  )

  def _rfDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "numTrees" -> Tuple2(50.0, 1000.0),
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "subSamplingRate" -> Tuple2(0.5, 1.0)
  )

  def _rfDefaultStringBoundaries = Map(
    "impurity" -> List("gini", "entropy"),
    "featureSubsetStrategy" -> List("all", "sqrt", "log2", "onethird")
  )

  def _treesDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "minInstancesPerNode" -> Tuple2(1.0, 50.0)
  )

  def _treesDefaultStringBoundaries: Map[String, List[String]] = Map(
    "impurity" -> List("gini", "entropy")
  )

  def _mlpcDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "layers" -> Tuple2(1.0, 10.0),
    "maxIter" -> Tuple2(10.0, 100.0),
    "stepSize" -> Tuple2(0.01, 1.0),
    "tol" -> Tuple2(1E-9, 1E-5),
    "hiddenLayerSizeAdjust" -> Tuple2(0.0, 50.0)
  )

  def _mlpcDefaultStringBoundaries: Map[String, List[String]] = Map(
    "solver" -> List("gd", "l-bfgs")
  )

  def _gbtDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxIter" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "minInstancesPerNode" -> Tuple2(1.0, 50.0),
    "stepSize" -> Tuple2(1E-4, 1.0)
  )

  def _gbtDefaultStringBoundaries: Map[String, List[String]] = Map(
    "impurity" -> List("gini", "entropy"),
    "lossType" -> List("logistic")
  )

  def _linearRegressionDefaultNumBoundaries: Map[String, (Double, Double)] = Map (
    "elasticNetParams" -> Tuple2(0.0, 1.0),
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tol" -> Tuple2(1E-9, 1E-5)
  )
  def _linearRegressionDefaultStringBoundaries: Map[String, List[String]] = Map (
    "loss" -> List("squaredError", "huber")
  )
  def _logisticRegressionDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "elasticNetParams" -> Tuple2(0.0, 1.0),
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tol" -> Tuple2(1E-9, 1E-5)
  )
  def _logisticRegressionDefaultStringBoundaries: Map[String, List[String]] = Map(
    "" -> List("")
  )
  def _svmDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tol" -> Tuple2(1E-9, 1E-5)
  )
  def _svmDefaultStringBoundaries: Map[String, List[String]] = Map(
    "" -> List("")
  )

  def _scoringDefaultClassifier = "f1"
  def _scoringOptimizationStrategyClassifier = "maximize"
  def _scoringDefaultRegressor = "rmse"
  def _scoringOptimizationStrategyRegressor = "minimize"

  def _modelTypeDefault = "RandomForest"

  def _mlFlowConfigDefaults: MLFlowConfig = MLFlowConfig(
    mlFlowTrackingURI = "hosted",
    mlFlowExperimentName = "default",
    mlFlowAPIToken = "default",
    mlFlowModelSaveDirectory = "s3://mlflow/experiments/"
  )

  def _defaultMlFlowLoggingFlag: Boolean = true

  def _mainConfigDefaults = MainConfig(
    modelFamily = _modelTypeDefault,
    labelCol = "label",
    featuresCol = "features",
    naFillFlag = true,
    varianceFilterFlag = true,
    outlierFilterFlag = true,
    pearsonFilteringFlag = true,
    covarianceFilteringFlag = true,
    scalingFlag = false,
    dateTimeConversionType = "split",
    fieldsToIgnoreInVector = _defaultFieldsToIgnoreInVector,
    numericBoundaries = _rfDefaultNumBoundaries,
    stringBoundaries = _rfDefaultStringBoundaries,
    scoringMetric = _scoringDefaultClassifier,
    scoringOptimizationStrategy = _scoringOptimizationStrategyClassifier,
    fillConfig = _fillConfigDefaults,
    outlierConfig = _outlierConfigDefaults,
    pearsonConfig = _pearsonConfigDefaults,
    covarianceConfig = _covarianceConfigDefaults,
    scalingConfig = _scalingConfigDefaults,
    geneticConfig = _geneticTunerDefaults,
    mlFlowLoggingFlag = _defaultMlFlowLoggingFlag,
    mlFlowConfig = _mlFlowConfigDefaults
  )

  def _featureImportancesDefaults = MainConfig(
    modelFamily = "RandomForest",
    labelCol = "label",
    featuresCol = "features",
    naFillFlag = true,
    varianceFilterFlag = true,
    outlierFilterFlag = false,
    pearsonFilteringFlag = false,
    covarianceFilteringFlag = false,
    scalingFlag = false,
    dateTimeConversionType = "split",
    fieldsToIgnoreInVector = _defaultFieldsToIgnoreInVector,
    numericBoundaries = _rfDefaultNumBoundaries,
    stringBoundaries = _rfDefaultStringBoundaries,
    scoringMetric = _scoringDefaultClassifier,
    scoringOptimizationStrategy = _scoringOptimizationStrategyClassifier,
    fillConfig = _fillConfigDefaults,
    outlierConfig = _outlierConfigDefaults,
    pearsonConfig = _pearsonConfigDefaults,
    covarianceConfig = _covarianceConfigDefaults,
    scalingConfig = _scalingConfigDefaults,
    geneticConfig = GeneticConfig(
      parallelism = 20,
      kFold = 1,
      trainPortion = 0.8,
      seed = 42L,
      firstGenerationGenePool = 25,
      numberOfGenerations = 20,
      numberOfParentsToRetain = 2,
      numberOfMutationsPerGeneration = 10,
      geneticMixing = 0.7,
      generationalMutationStrategy = "linear",
      fixedMutationValue = 1,
      mutationMagnitudeMode = "fixed"
    ),
    mlFlowLoggingFlag = _defaultMlFlowLoggingFlag,
    mlFlowConfig = _mlFlowConfigDefaults
  )

  def _treeSplitDefaults =  MainConfig(
    modelFamily = "Trees",
    labelCol = "label",
    featuresCol = "features",
    naFillFlag = true,
    varianceFilterFlag = true,
    outlierFilterFlag = false,
    pearsonFilteringFlag = false,
    covarianceFilteringFlag = false,
    scalingFlag = false,
    dateTimeConversionType = "split",
    fieldsToIgnoreInVector = _defaultFieldsToIgnoreInVector,
    numericBoundaries = _treesDefaultNumBoundaries,
    stringBoundaries = _treesDefaultStringBoundaries,
    scoringMetric = _scoringDefaultClassifier,
    scoringOptimizationStrategy = _scoringOptimizationStrategyClassifier,
    fillConfig = _fillConfigDefaults,
    outlierConfig = _outlierConfigDefaults,
    pearsonConfig = _pearsonConfigDefaults,
    covarianceConfig = _covarianceConfigDefaults,
    scalingConfig = _scalingConfigDefaults,
    geneticConfig = GeneticConfig(
      parallelism = 20,
      kFold = 1,
      trainPortion = 0.8,
      seed = 42L,
      firstGenerationGenePool = 25,
      numberOfGenerations = 20,
      numberOfParentsToRetain = 2,
      numberOfMutationsPerGeneration = 10,
      geneticMixing = 0.7,
      generationalMutationStrategy = "linear",
      fixedMutationValue = 1,
      mutationMagnitudeMode = "fixed"
    ),
    mlFlowLoggingFlag = _defaultMlFlowLoggingFlag,
    mlFlowConfig = _mlFlowConfigDefaults
  )

}