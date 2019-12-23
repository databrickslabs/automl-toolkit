package com.databricks.labs.automl.params

import com.databricks.labs.automl.utils.InitDbUtils

trait Defaults {

  final val _supportedModels: Array[String] = Array(
    "GBT",
    "Trees",
    "RandomForest",
    "LinearRegression",
    "LogisticRegression",
    "MLPC",
    "SVM",
    "XGBoost",
    "gbmBinary",
    "gbmMulti",
    "gbmMultiOVA",
    "gbmHuber",
    "gbmFair",
    "gbmLasso",
    "gbmRidge",
    "gbmPoisson",
    "gbmQuantile",
    "gbmMape",
    "gbmTweedie",
    "gbmGamma"
  )

  final val trainSplitMethods: List[String] = List(
    "random",
    "chronological",
    "stratifyReduce",
    "stratified",
    "overSample",
    "underSample",
    "kSample"
  )

  final val _supportedFeatureImportanceCutoffTypes: List[String] =
    List("none", "value", "count")

  final val _allowableEvolutionStrategies = List("batch", "continuous")

  final val _allowableMlFlowLoggingModes =
    List("tuningOnly", "bestOnly", "full")

  final val _allowableInitialGenerationModes = List("random", "permutations")

  final val _allowableInitialGenerationIndexMixingModes =
    List("random", "linear")

  final val allowableKMeansDistanceMeasurements: List[String] =
    List("cosine", "euclidean")
  final val allowableMutationModes: List[String] =
    List("weighted", "random", "ratio")
  final val allowableVectorMutationMethods: List[String] =
    List("random", "fixed", "all")
  final val allowableLabelBalanceModes: List[String] =
    List("match", "percentage", "target")

  final val allowableDateTimeConversions = List("unix", "split")
  final val allowableCategoricalFilterModes = List("silent", "warn")
  final val allowableCardinalilties = List("approx", "exact")
  final val _allowableNAFillModes: List[String] =
    List(
      "auto",
      "mapFill",
      "blanketFillAll",
      "blanketFillCharOnly",
      "blanketFillNumOnly"
    )

  final val allowableMBORegressorTypes =
    List("XGBoost", "LinearRegression", "RandomForest")

  final val allowableFeatureInteractionModes =
    List("optimistic", "strict", "all")

  def _defaultModelingFamily: String = "RandomForest"

  def _defaultLabelCol: String = "label"

  def _defaultFeaturesCol: String = "features"

  def _defaultNAFillFlag: Boolean = true

  def _defaultVarianceFilterFlag: Boolean = true

  def _defaultOutlierFilterFlag: Boolean = false

  def _defaultPearsonFilterFlag: Boolean = false

  def _defaultCovarianceFilterFlag: Boolean = false

  def _defaultOneHotEncodeFlag: Boolean = false

  def _defaultScalingFlag: Boolean = false

  def _defaultFeatureInteractionFlag: Boolean = false

  def _defaultDataPrepCachingFlag: Boolean = true

  def _defaultDataReductionFactor: Double = 0.5

  def _defaultPipelineDebugFlag: Boolean = false

  def _defaultDateTimeConversionType: String = "split"

  def _defaultFieldsToIgnoreInVector: Array[String] = Array.empty[String]

  def _defaultHyperSpaceInference: Boolean = false

  def _defaultHyperSpaceInferenceCount: Int = 200000

  def _defaultHyperSpaceModelType: String = "RandomForest"

  def _defaultHyperSpaceModelCount: Int = 10

  def _defaultInitialGenerationMode: String = "random"

  def _defaultDataPrepParallelism: Int = 20

  def _defaultFirstGenerationConfig = FirstGenerationConfig(
    permutationCount = 10,
    indexMixingMode = "linear",
    arraySeed = 42L
  )

  def _defaultFeatureInteractionConfig = FeatureInteractionConfig(
    retentionMode = "optimistic",
    continuousDiscretizerBucketCount = 10,
    parallelism = 12,
    targetInteractionPercentage = 10
  )

  def _defaultKSampleConfig: KSampleConfig = KSampleConfig(
    syntheticCol = "synthetic_kSample",
    kGroups = 25,
    kMeansMaxIter = 100,
    kMeansTolerance = 1E-6,
    kMeansDistanceMeasurement = "euclidean",
    kMeansSeed = 42L,
    kMeansPredictionCol = "kGroups_kSample",
    lshHashTables = 10,
    lshSeed = 42L,
    lshOutputCol = "hashes_kSample",
    quorumCount = 7,
    minimumVectorCountToMutate = 1,
    vectorMutationMethod = "random",
    mutationMode = "weighted",
    mutationValue = 0.5,
    labelBalanceMode = "percentage",
    cardinalityThreshold = 20,
    numericRatio = 0.2,
    numericTarget = 500,
    outputDfRepartitionScaleFactor = 3
  )

  def _geneticTunerDefaults = GeneticConfig(
    parallelism = 20,
    kFold = 5,
    trainPortion = 0.8,
    trainSplitMethod = "random",
    kSampleConfig = _defaultKSampleConfig,
    trainSplitChronologicalColumn = "datetime",
    trainSplitChronologicalRandomPercentage = 0.0,
    seed = 42L,
    firstGenerationGenePool = 20,
    numberOfGenerations = 10,
    numberOfParentsToRetain = 3,
    numberOfMutationsPerGeneration = 10,
    geneticMixing = 0.7,
    generationalMutationStrategy = "linear",
    fixedMutationValue = 1,
    mutationMagnitudeMode = "fixed",
    evolutionStrategy = "batch",
    geneticMBORegressorType = "XGBoost",
    geneticMBOCandidateFactor = 10,
    continuousEvolutionMaxIterations = 200,
    continuousEvolutionStoppingScore = 1.0,
    continuousEvolutionImprovementThreshold = -10,
    continuousEvolutionParallelism = 4,
    continuousEvolutionMutationAggressiveness = 3,
    continuousEvolutionGeneticMixing = 0.7,
    continuousEvolutionRollingImprovementCount = 20,
    modelSeed = Map.empty,
    hyperSpaceInference = _defaultHyperSpaceInference,
    hyperSpaceInferenceCount = _defaultHyperSpaceInferenceCount,
    hyperSpaceModelCount = _defaultHyperSpaceModelCount,
    hyperSpaceModelType = _defaultHyperSpaceModelType,
    initialGenerationMode = _defaultInitialGenerationMode,
    initialGenerationConfig = _defaultFirstGenerationConfig
  )

  def _fillConfigDefaults = FillConfig(
    numericFillStat = "mean",
    characterFillStat = "max",
    modelSelectionDistinctThreshold = 10,
    cardinalitySwitch = true,
    cardinalityType = "exact",
    cardinalityLimit = 200,
    cardinalityPrecision = 0.05,
    cardinalityCheckMode = "silent",
    filterPrecision = 0.01,
    categoricalNAFillMap = Map.empty[String, String],
    numericNAFillMap = Map.empty[String, AnyVal],
    characterNABlanketFillValue = "",
    numericNABlanketFillValue = 0.0,
    naFillMode = "auto"
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
    filterManualValue = 1.0,
    filterMode = "auto",
    autoFilterNTile = 0.99
  )

  def _covarianceConfigDefaults =
    CovarianceConfig(correlationCutoffLow = -0.99, correlationCutoffHigh = 0.99)

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

  def _xgboostDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "alpha" -> Tuple2(0.0, 1.0),
    "eta" -> Tuple2(0.1, 0.5),
    "gamma" -> Tuple2(0.0, 10.0),
    "lambda" -> Tuple2(0.1, 10.0),
    "maxDepth" -> Tuple2(3.0, 10.0),
    "subSample" -> Tuple2(0.4, 0.6),
    "minChildWeight" -> Tuple2(0.1, 10.0),
    "numRound" -> Tuple2(25.0, 250.0),
    "maxBins" -> Tuple2(25.0, 512.0),
    "trainTestRatio" -> Tuple2(0.2, 0.8)
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
    "featureSubsetStrategy" -> List("auto")
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
    "tolerance" -> Tuple2(1E-9, 1E-5),
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

  def _gbtDefaultStringBoundaries: Map[String, List[String]] =
    Map("impurity" -> List("gini", "entropy"), "lossType" -> List("logistic"))

  def _linearRegressionDefaultNumBoundaries: Map[String, (Double, Double)] =
    Map(
      "elasticNetParams" -> Tuple2(0.0, 1.0),
      "maxIter" -> Tuple2(100.0, 10000.0),
      "regParam" -> Tuple2(0.0, 1.0),
      "tolerance" -> Tuple2(1E-9, 1E-5)
    )
  def _linearRegressionDefaultStringBoundaries: Map[String, List[String]] = Map(
    "loss" -> List("squaredError", "huber")
  )
  def _logisticRegressionDefaultNumBoundaries: Map[String, (Double, Double)] =
    Map(
      "elasticNetParams" -> Tuple2(0.0, 1.0),
      "maxIter" -> Tuple2(100.0, 10000.0),
      "regParam" -> Tuple2(0.0, 1.0),
      "tolerance" -> Tuple2(1E-9, 1E-5)
    )
  def _logisticRegressionDefaultStringBoundaries: Map[String, List[String]] =
    Map("" -> List(""))
  def _svmDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5)
  )
  def _svmDefaultStringBoundaries: Map[String, List[String]] = Map(
    "" -> List("")
  )

  def _naiveBayesDefaultStringBoundaries: Map[String, List[String]] = Map(
    "modelType" -> List("multinomial", "bernoulli")
  )

  def _naiveBayesDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "smoothing" -> Tuple2(0.0, 1.0)
  )

  def _lightGBMDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "baggingFraction" -> Tuple2(0.5, 1.0),
    "baggingFreq" -> Tuple2(0.0, 1.0),
    "featureFraction" -> Tuple2(0.6, 1.0),
    "learningRate" -> Tuple2(1E-8, 1.0),
    "maxBin" -> Tuple2(50, 1000),
    "maxDepth" -> Tuple2(3.0, 20.0),
    "minSumHessianInLeaf" -> Tuple2(1e-5, 50.0),
    "numIterations" -> Tuple2(25.0, 250.0),
    "numLeaves" -> Tuple2(10.0, 50.0),
    "lambdaL1" -> Tuple2(0.0, 1.0),
    "lambdaL2" -> Tuple2(0.0, 1.0),
    "alpha" -> Tuple2(0.0, 1.0)
  )

  def _lightGBMDefaultStringBoundaries: Map[String, List[String]] = Map(
    "boostingType" -> List("gbdt", "rf", "dart", "goss")
  )

  def _scoringDefaultClassifier = "f1"
  def _scoringOptimizationStrategyClassifier = "maximize"
  def _scoringDefaultRegressor = "rmse"
  def _scoringOptimizationStrategyRegressor = "minimize"

  def _modelTypeDefault = "RandomForest"

  def _mlFlowConfigDefaults: MLFlowConfig = {
    val mlfloWLoggingConfig =
      InitDbUtils.getMlFlowLoggingConfig(_defaultMlFlowLoggingFlag)
    MLFlowConfig(
      mlFlowTrackingURI = mlfloWLoggingConfig.mlFlowTrackingURI,
      mlFlowExperimentName = mlfloWLoggingConfig.mlFlowExperimentName,
      mlFlowAPIToken = mlfloWLoggingConfig.mlFlowAPIToken,
      mlFlowModelSaveDirectory = mlfloWLoggingConfig.mlFlowModelSaveDirectory,
      mlFlowLoggingMode = "full",
      mlFlowBestSuffix = "_best",
      mlFlowCustomRunTags = Map[String, String]()
    )
  }

  def _inferenceConfigSaveLocationDefault: String = "/models"

  def _defaultMlFlowLoggingFlag: Boolean = false

  def _defaultMlFlowArtifactsFlag: Boolean = false

  def _defaultAutoStoppingFlag: Boolean = true

  def _defaultAutoStoppingScore: Double = 0.95

  def _defaultFeatureImportanceCutoffType: String = "count"

  def _defaultFeatureImportanceCutoffValue: Double = 15.0

  def _mainConfigDefaults = MainConfig(
    modelFamily = _modelTypeDefault,
    labelCol = "label",
    featuresCol = "features",
    naFillFlag = true,
    varianceFilterFlag = true,
    outlierFilterFlag = false,
    pearsonFilteringFlag = false,
    covarianceFilteringFlag = false,
    oneHotEncodeFlag = false,
    scalingFlag = false,
    featureInteractionFlag = false,
    dataPrepCachingFlag = true,
    autoStoppingFlag = _defaultAutoStoppingFlag,
    dataPrepParallelism = _defaultDataPrepParallelism,
    autoStoppingScore = _defaultAutoStoppingScore,
    featureImportanceCutoffType = _defaultFeatureImportanceCutoffType,
    featureImportanceCutoffValue = _defaultFeatureImportanceCutoffValue,
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
    featureInteractionConfig = _defaultFeatureInteractionConfig,
    scalingConfig = _scalingConfigDefaults,
    geneticConfig = _geneticTunerDefaults,
    mlFlowLoggingFlag = _defaultMlFlowLoggingFlag,
    mlFlowLogArtifactsFlag = _defaultMlFlowArtifactsFlag,
    mlFlowConfig = _mlFlowConfigDefaults,
    inferenceConfigSaveLocation = _inferenceConfigSaveLocationDefault,
    dataReductionFactor = _defaultDataReductionFactor,
    pipelineDebugFlag = _defaultPipelineDebugFlag
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
    oneHotEncodeFlag = false,
    scalingFlag = false,
    featureInteractionFlag = false,
    dataPrepCachingFlag = true,
    autoStoppingFlag = _defaultAutoStoppingFlag,
    dataPrepParallelism = _defaultDataPrepParallelism,
    autoStoppingScore = _defaultAutoStoppingScore,
    featureImportanceCutoffType = _defaultFeatureImportanceCutoffType,
    featureImportanceCutoffValue = _defaultFeatureImportanceCutoffValue,
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
    featureInteractionConfig = _defaultFeatureInteractionConfig,
    geneticConfig = GeneticConfig(
      parallelism = 20,
      kFold = 1,
      trainPortion = 0.8,
      trainSplitMethod = "random",
      kSampleConfig = _defaultKSampleConfig,
      trainSplitChronologicalColumn = "datetime",
      trainSplitChronologicalRandomPercentage = 0.0,
      seed = 42L,
      firstGenerationGenePool = 25,
      numberOfGenerations = 20,
      numberOfParentsToRetain = 2,
      numberOfMutationsPerGeneration = 10,
      geneticMixing = 0.7,
      generationalMutationStrategy = "linear",
      fixedMutationValue = 1,
      mutationMagnitudeMode = "fixed",
      evolutionStrategy = "batch",
      geneticMBORegressorType = "XGBoost",
      geneticMBOCandidateFactor = 10,
      continuousEvolutionMaxIterations = 200,
      continuousEvolutionStoppingScore = 1.0,
      continuousEvolutionImprovementThreshold = -10,
      continuousEvolutionParallelism = 4,
      continuousEvolutionMutationAggressiveness = 3,
      continuousEvolutionGeneticMixing = 0.7,
      continuousEvolutionRollingImprovementCount = 20,
      modelSeed = Map.empty,
      hyperSpaceInference = _defaultHyperSpaceInference,
      hyperSpaceInferenceCount = _defaultHyperSpaceInferenceCount,
      hyperSpaceModelType = _defaultHyperSpaceModelType,
      hyperSpaceModelCount = _defaultHyperSpaceModelCount,
      initialGenerationMode = _defaultInitialGenerationMode,
      initialGenerationConfig = _defaultFirstGenerationConfig
    ),
    mlFlowLoggingFlag = _defaultMlFlowLoggingFlag,
    mlFlowLogArtifactsFlag = _defaultMlFlowArtifactsFlag,
    mlFlowConfig = _mlFlowConfigDefaults,
    inferenceConfigSaveLocation = _inferenceConfigSaveLocationDefault,
    dataReductionFactor = _defaultDataReductionFactor,
    pipelineDebugFlag = false
  )

  def _treeSplitDefaults = MainConfig(
    modelFamily = "Trees",
    labelCol = "label",
    featuresCol = "features",
    naFillFlag = true,
    varianceFilterFlag = true,
    outlierFilterFlag = false,
    pearsonFilteringFlag = false,
    covarianceFilteringFlag = false,
    oneHotEncodeFlag = false,
    scalingFlag = false,
    featureInteractionFlag = false,
    dataPrepCachingFlag = true,
    dateTimeConversionType = "split",
    autoStoppingFlag = _defaultAutoStoppingFlag,
    dataPrepParallelism = _defaultDataPrepParallelism,
    autoStoppingScore = _defaultAutoStoppingScore,
    featureImportanceCutoffType = _defaultFeatureImportanceCutoffType,
    featureImportanceCutoffValue = _defaultFeatureImportanceCutoffValue,
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
    featureInteractionConfig = _defaultFeatureInteractionConfig,
    geneticConfig = GeneticConfig(
      parallelism = 20,
      kFold = 1,
      trainPortion = 0.8,
      trainSplitMethod = "random",
      kSampleConfig = _defaultKSampleConfig,
      trainSplitChronologicalColumn = "datetime",
      trainSplitChronologicalRandomPercentage = 0.0,
      seed = 42L,
      firstGenerationGenePool = 25,
      numberOfGenerations = 20,
      numberOfParentsToRetain = 2,
      numberOfMutationsPerGeneration = 10,
      geneticMixing = 0.7,
      generationalMutationStrategy = "linear",
      fixedMutationValue = 1,
      mutationMagnitudeMode = "fixed",
      evolutionStrategy = "batch",
      geneticMBORegressorType = "XGBoost",
      geneticMBOCandidateFactor = 10,
      continuousEvolutionMaxIterations = 200,
      continuousEvolutionStoppingScore = 1.0,
      continuousEvolutionImprovementThreshold = -10,
      continuousEvolutionParallelism = 4,
      continuousEvolutionMutationAggressiveness = 3,
      continuousEvolutionGeneticMixing = 0.7,
      continuousEvolutionRollingImprovementCount = 20,
      modelSeed = Map.empty,
      hyperSpaceInference = _defaultHyperSpaceInference,
      hyperSpaceInferenceCount = _defaultHyperSpaceInferenceCount,
      hyperSpaceModelType = _defaultHyperSpaceModelType,
      hyperSpaceModelCount = _defaultHyperSpaceModelCount,
      initialGenerationMode = _defaultInitialGenerationMode,
      initialGenerationConfig = _defaultFirstGenerationConfig
    ),
    mlFlowLoggingFlag = _defaultMlFlowLoggingFlag,
    mlFlowLogArtifactsFlag = _defaultMlFlowArtifactsFlag,
    mlFlowConfig = _mlFlowConfigDefaults,
    inferenceConfigSaveLocation = _inferenceConfigSaveLocationDefault,
    dataReductionFactor = _defaultDataReductionFactor,
    pipelineDebugFlag = false
  )
}
