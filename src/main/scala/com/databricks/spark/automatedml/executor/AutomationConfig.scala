package com.databricks.spark.automatedml.executor

import com.databricks.spark.automatedml.params._
import com.databricks.spark.automatedml.sanitize.SanitizerDefaults

trait AutomationConfig extends Defaults with SanitizerDefaults {


  var _modelingFamily: String = _defaultModelingFamily

  var _labelCol: String = _defaultLabelCol

  var _featuresCol: String = _defaultFeaturesCol

  var _naFillFlag: Boolean = _defaultNAFillFlag

  var _varianceFilterFlag: Boolean = _defaultVarianceFilterFlag

  var _outlierFilterFlag: Boolean = _defaultOutlierFilterFlag

  var _pearsonFilterFlag: Boolean = _defaultPearsonFilterFlag

  var _covarianceFilterFlag: Boolean = _defaultCovarianceFilterFlag

  var _scalingFlag: Boolean = _defaultScalingFlag

  var _numericBoundaries: Map[String, (Double, Double)] = _rfDefaultNumBoundaries

  var _stringBoundaries: Map[String, List[String]] = _rfDefaultStringBoundaries

  var _scoringMetric: String = _scoringDefaultClassifier

  var _scoringOptimizationStrategy: String = _scoringOptimizationStrategyClassifier

  var _numericFillStat: String = _fillConfigDefaults.numericFillStat

  var _characterFillStat: String = _fillConfigDefaults.characterFillStat

  var _dateTimeConversionType: String = _defaultDateTimeConversionType

  var _fieldsToIgnoreInVector: Array[String] = _defaultFieldsToIgnoreInVector

  var _modelSelectionDistinctThreshold: Int = _fillConfigDefaults.modelSelectionDistinctThreshold

  var _fillConfig: FillConfig = _fillConfigDefaults

  var _filterBounds: String = _outlierConfigDefaults.filterBounds

  var _lowerFilterNTile: Double = _outlierConfigDefaults.lowerFilterNTile

  var _upperFilterNTile: Double = _outlierConfigDefaults.upperFilterNTile

  var _filterPrecision: Double = _outlierConfigDefaults.filterPrecision

  var _continuousDataThreshold: Int = _outlierConfigDefaults.continuousDataThreshold

  var _fieldsToIgnore: Array[String] = _outlierConfigDefaults.fieldsToIgnore

  var _outlierConfig: OutlierConfig = _outlierConfigDefaults

  var _pearsonFilterStatistic: String = _pearsonConfigDefaults.filterStatistic

  var _pearsonFilterDirection: String = _pearsonConfigDefaults.filterDirection

  var _pearsonFilterManualValue: Double = _pearsonConfigDefaults.filterManualValue

  var _pearsonFilterMode: String = _pearsonConfigDefaults.filterMode

  var _pearsonAutoFilterNTile: Double = _pearsonConfigDefaults.autoFilterNTile

  var _pearsonConfig: PearsonConfig = _pearsonConfigDefaults

  var _correlationCutoffLow: Double = _covarianceConfigDefaults.correlationCutoffLow

  var _correlationCutoffHigh: Double = _covarianceConfigDefaults.correlationCutoffHigh

  var _covarianceConfig: CovarianceConfig = _covarianceConfigDefaults

  var _scalerType: String = defaultScalerType

  var _scalerMin: Double = defaultScalerMin

  var _scalerMax: Double = defaultScalerMax

  var _standardScalerMeanFlag: Boolean = defaultStandardScalerMeanFlag

  var _standardScalerStdDevFlag: Boolean = defaultStandardScalerStdDevFlag

  var _pNorm: Double = defaultPNorm

  var _scalingConfig: ScalingConfig = _scalingConfigDefaults

  var _parallelism: Int = _geneticTunerDefaults.parallelism

  var _kFold: Int = _geneticTunerDefaults.kFold

  var _trainPortion: Double = _geneticTunerDefaults.trainPortion

  var _trainSplitMethod: String = _geneticTunerDefaults.trainSplitMethod

  var _trainSplitChronologicalColumn: String = _geneticTunerDefaults.trainSplitChronologicalColumn

  var _trainSplitChronologicalRandomPercentage: Double = _geneticTunerDefaults.trainSplitChronologicalRandomPercentage

  var _trainSplitColumnSet: Boolean = false

  var _seed: Long = _geneticTunerDefaults.seed

  var _firstGenerationGenePool: Int = _geneticTunerDefaults.firstGenerationGenePool

  var _numberOfGenerations: Int = _geneticTunerDefaults.numberOfGenerations

  var _numberOfParentsToRetain: Int = _geneticTunerDefaults.numberOfParentsToRetain

  var _numberOfMutationsPerGeneration: Int = _geneticTunerDefaults.numberOfMutationsPerGeneration

  var _geneticMixing: Double = _geneticTunerDefaults.geneticMixing

  var _generationalMutationStrategy: String = _geneticTunerDefaults.generationalMutationStrategy

  var _fixedMutationValue: Int = _geneticTunerDefaults.fixedMutationValue

  var _mutationMagnitudeMode: String = _geneticTunerDefaults.mutationMagnitudeMode

  var _geneticConfig: GeneticConfig = _geneticTunerDefaults

  var _mainConfig: MainConfig = _mainConfigDefaults

  var _featureImportancesConfig: MainConfig = _featureImportancesDefaults

  var _treeSplitsConfig: MainConfig = _treeSplitDefaults

  var _mlFlowConfig: MLFlowConfig = _mlFlowConfigDefaults

  var _mlFlowLoggingFlag: Boolean = _defaultMlFlowLoggingFlag

  var _mlFlowTrackingURI: String = _mlFlowConfigDefaults.mlFlowTrackingURI

  var _mlFlowExperimentName: String = _mlFlowConfigDefaults.mlFlowExperimentName

  var _mlFlowAPIToken: String = _mlFlowConfigDefaults.mlFlowAPIToken

  var _mlFlowModelSaveDirectory: String = _mlFlowConfigDefaults.mlFlowModelSaveDirectory

  var _autoStoppingFlag: Boolean = _defaultAutoStoppingFlag

  var _autoStoppingScore: Double = _defaultAutoStoppingScore

  var _featureImportanceCutoffType: String = _defaultFeatureImportanceCutoffType

  var _featureImportanceCutoffValue: Double = _defaultFeatureImportanceCutoffValue

  var _evolutionStrategy: String = _geneticTunerDefaults.evolutionStrategy

  var _continuousEvolutionMaxIterations: Int = _geneticTunerDefaults.continuousEvolutionMaxIterations

  var _continuousEvolutionStoppingScore: Double = _geneticTunerDefaults.continuousEvolutionStoppingScore

  var _continuousEvolutionParallelism: Int = _geneticTunerDefaults.continuousEvolutionParallelism

  var _continuousEvolutionMutationAggressiveness: Int = _geneticTunerDefaults.continuousEvolutionMutationAggressiveness

  var _continuousEvolutionGeneticMixing: Double = _geneticTunerDefaults.continuousEvolutionGeneticMixing

  var _continuousEvolutionRollingImprovementCount: Int = _geneticTunerDefaults.continuousEvolutionRollingImprovementCount

  private def setConfigs(): this.type = {
    setMainConfig()
    setTreeSplitsConfig()
    setFeatConfig()
  }

  def setModelingFamily(value: String): this.type = {
    _modelingFamily = value
    _numericBoundaries = value match {
      case "RandomForest" => _rfDefaultNumBoundaries
      case "MLPC" => _mlpcDefaultNumBoundaries
      case "Trees" => _treesDefaultNumBoundaries
      case "GBT" => _gbtDefaultNumBoundaries
      case "LinearRegression" => _linearRegressionDefaultNumBoundaries
      case "LogisticRegression" => _logisticRegressionDefaultNumBoundaries
      case "SVM" => _svmDefaultNumBoundaries
      case _ => throw new IllegalArgumentException(s"$value is an unsupported Model Type")
    }
    _stringBoundaries = value match {
      case "RandomForest" => _rfDefaultStringBoundaries
      case "MLPC" => _mlpcDefaultStringBoundaries
      case "Trees" => _treesDefaultStringBoundaries
      case "GBT" => _gbtDefaultStringBoundaries
      case "LinearRegression" => _linearRegressionDefaultStringBoundaries
      case "LogisticRegression" => _logisticRegressionDefaultStringBoundaries
      case "SVM" => _svmDefaultStringBoundaries
      case _ => throw new IllegalArgumentException(s"$value is an unsupported Model Type")
    }
    setConfigs()
    this
  }

  def setLabelCol(value: String): this.type = {
    _labelCol = value
    setConfigs()
    this
  }

  def setFeaturesCol(value: String): this.type = {
    _featuresCol = value
    setConfigs()
    this
  }

  def naFillOn(): this.type = {
    _naFillFlag = true
    setConfigs()
    this
  }

  def naFillOff(): this.type = {
    _naFillFlag = false
    setConfigs()
    this
  }

  def varianceFilterOn(): this.type = {
    _varianceFilterFlag = true
    setConfigs()
    this
  }

  def varianceFilterOff(): this.type = {
    _varianceFilterFlag = false
    setConfigs()
    this
  }

  def outlierFilterOn(): this.type = {
    _outlierFilterFlag = true
    setConfigs()
    this
  }

  def outlierFilterOff(): this.type = {
    _outlierFilterFlag = false
    setConfigs()
    this
  }

  def pearsonFilterOn(): this.type = {
    _pearsonFilterFlag = true
    setConfigs()
    this
  }

  def pearsonFilterOff(): this.type = {
    _pearsonFilterFlag = false
    setConfigs()
    this
  }

  def covarianceFilterOn(): this.type = {
    _covarianceFilterFlag = true
    setConfigs()
    this
  }

  def covarianceFilterOff(): this.type = {
    _covarianceFilterFlag = false
    setConfigs()
    this
  }

  def scalingOn(): this.type = {
    _scalingFlag = true
    setConfigs()
    this
  }

  def scalingOff(): this.type = {
    _scalingFlag = false
    setConfigs()
    this
  }

  def setNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    _numericBoundaries = value
    setConfigs()
    this
  }

  def setStringBoundaries(value: Map[String, List[String]]): this.type = {
    _stringBoundaries = value
    setConfigs()
    this
  }

  def setNumericFillStat(value: String): this.type = {
    _numericFillStat = value
    setFillConfig()
    setConfigs()
    this
  }

  def setCharacterFillStat(value: String): this.type = {
    _characterFillStat = value
    setFillConfig()
    setConfigs()
    this
  }

  def setDateTimeConversionType(value: String): this.type = {
    _dateTimeConversionType = value
    setConfigs()
    this
  }

  def setFieldsToIgnoreInVector(value: Array[String]): this.type = {
    _fieldsToIgnoreInVector = value
    if (_trainSplitColumnSet)
      _fieldsToIgnoreInVector = _fieldsToIgnoreInVector :+ _trainSplitChronologicalColumn
    setConfigs()
    this
  }

  def setModelSelectionDistinctThreshold(value: Int): this.type = {
    _modelSelectionDistinctThreshold = value
    setFillConfig()
    setConfigs()
    this
  }

  private def setFillConfig(): this.type = {
    _fillConfig = FillConfig(
      numericFillStat = _numericFillStat,
      characterFillStat = _characterFillStat,
      modelSelectionDistinctThreshold = _modelSelectionDistinctThreshold
    )
    this
  }

  def setFilterBounds(value: String): this.type = {
    _filterBounds = value
    setOutlierConfig()
    setConfigs()
    this
  }

  def setLowerFilterNTile(value: Double): this.type = {
    _lowerFilterNTile = value
    setOutlierConfig()
    setConfigs()
    this
  }

  def setUpperFilterNTile(value: Double): this.type = {
    _upperFilterNTile = value
    setOutlierConfig()
    setConfigs()
    this
  }

  def setFilterPrecision(value: Double): this.type = {
    _filterPrecision = value
    setOutlierConfig()
    setConfigs()
    this
  }

  def setContinuousDataThreshold(value: Int): this.type = {
    _continuousDataThreshold = value
    setOutlierConfig()
    setConfigs()
    this
  }

  def setFieldsToIgnore(value: Array[String]): this.type = {
    _fieldsToIgnore = value
    setOutlierConfig()
    setConfigs()
    this
  }

  private def setOutlierConfig(): this.type = {
    _outlierConfig = OutlierConfig(
      filterBounds = _filterBounds,
      lowerFilterNTile = _lowerFilterNTile,
      upperFilterNTile = _upperFilterNTile,
      filterPrecision = _filterPrecision,
      continuousDataThreshold = _continuousDataThreshold,
      fieldsToIgnore = _fieldsToIgnore
    )
    this
  }

  def setPearsonFilterStatistic(value: String): this.type = {
    _pearsonFilterStatistic = value
    setPearsonConfig()
    setConfigs()
    this
  }

  def setPearsonFilterDirection(value: String): this.type = {
    _pearsonFilterDirection = value
    setPearsonConfig()
    setConfigs()
    this
  }

  def setPearsonFilterManualValue(value: Double): this.type = {
    _pearsonFilterManualValue = value
    setPearsonConfig()
    setConfigs()
    this
  }

  def setPearsonFilterMode(value: String): this.type = {
    _pearsonFilterMode = value
    setPearsonConfig()
    setConfigs()
    this
  }

  def setPearsonAutoFilterNTile(value: Double): this.type = {
    _pearsonAutoFilterNTile = value
    setPearsonConfig()
    setConfigs()
    this
  }

  private def setPearsonConfig(): this.type = {
    _pearsonConfig = PearsonConfig(
      filterStatistic = _pearsonFilterStatistic,
      filterDirection = _pearsonFilterDirection,
      filterManualValue = _pearsonFilterManualValue,
      filterMode = _pearsonFilterMode,
      autoFilterNTile = _pearsonAutoFilterNTile
    )
    this
  }

  def setCorrelationCutoffLow(value: Double): this.type = {
    _correlationCutoffLow = value
    setCovarianceConfig()
    setConfigs()
    this
  }

  def setCorrelationCutoffHigh(value: Double): this.type = {
    _correlationCutoffHigh = value
    setCovarianceConfig()
    setConfigs()
    this
  }

  private def setCovarianceConfig(): this.type = {
    _covarianceConfig = CovarianceConfig(
      correlationCutoffLow = _correlationCutoffLow,
      correlationCutoffHigh = _correlationCutoffHigh
    )
    this
  }

  def setScalerType(value: String): this.type = {
    _scalerType = value
    setScalerConfig()
    setConfigs()
    this
  }

  def setScalerMin(value: Double): this.type = {
    _scalerMin = value
    setScalerConfig()
    setConfigs()
    this
  }

  def setScalerMax(value: Double): this.type = {
    _scalerMax = value
    setScalerConfig()
    setConfigs()
    this
  }

  def setStandardScalerMeanFlagOn(): this.type = {
    _standardScalerMeanFlag = true
    setScalerConfig()
    setConfigs()
    this
  }

  def setStandardScalerMeanFlagOff(): this.type = {
    _standardScalerMeanFlag = false
    setScalerConfig()
    setConfigs()
    this
  }

  def setStandardScalerStdDevFlagOn(): this.type = {
    _standardScalerStdDevFlag = true
    setScalerConfig()
    setConfigs()
    this
  }

  def setStandardScalerStdDevFlagOff(): this.type = {
    _standardScalerStdDevFlag = false
    setScalerConfig()
    setConfigs()
    this
  }

  def setPNorm(value: Double): this.type = {
    _pNorm = value
    setScalerConfig()
    setConfigs()
    this
  }

  private def setScalerConfig(): this.type = {
    _scalingConfig = ScalingConfig(
      scalerType = _scalerType,
      scalerMin = _scalerMin,
      scalerMax = _scalerMax,
      standardScalerMeanFlag = _standardScalerMeanFlag,
      standardScalerStdDevFlag = _standardScalerStdDevFlag,
      pNorm = _pNorm
    )
    this
  }

  def setParallelism(value: Integer): this.type = {
    //TODO: FIND OUT WHAT THIS RESTRICTION NEEDS TO BE FOR PARALLELISM.
    require(_parallelism < 10000, s"Parallelism above 10000 will result in cluster instability.")
    _parallelism = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setKFold(value: Integer): this.type = {
    _kFold = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setTrainPortion(value: Double): this.type = {
    _trainPortion = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setTrainSplitMethod(value: String): this.type = {
    require(trainSplitMethods.contains(value),
      s"TrainSplitMethod $value must be one of: ${trainSplitMethods.mkString(", ")}")
    _trainSplitMethod = value
    if (value == "chronological")
      println("[WARNING] setTrainSplitMethod() -> Chronological splits is shuffle-intensive and will increase " +
        "runtime significantly.  Only use if necessary for modeling scenario!")
    setGeneticConfig()
    setConfigs()
    this
  }

  def setTrainSplitChronologicalColumn(value: String): this.type = {
    _trainSplitChronologicalColumn = value
    val ignoredFields: Array[String] = _fieldsToIgnoreInVector ++: Array(value)
    setFieldsToIgnoreInVector(ignoredFields)
    _trainSplitColumnSet = true
    setGeneticConfig()
    setConfigs()
    this
  }

  def setTrainSplitChronologicalRandomPercentage(value: Double): this.type = {
    _trainSplitChronologicalRandomPercentage = value
    if(value > 10) println("[WARNING] setTrainSplitChronologicalRandomPercentage() setting this value above 10 " +
      "percent will cause significant per-run train/test skew and variability in row counts during training.  " +
      "Use higher values only if this is desired.")
    setGeneticConfig()
    setConfigs()
    this
  }

  def setSeed(value: Long): this.type = {
    _seed = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setFirstGenerationGenePool(value: Int): this.type = {
    _firstGenerationGenePool = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setNumberOfGenerations(value: Int): this.type = {
    _numberOfGenerations = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setNumberOfParentsToRetain(value: Int): this.type = {
    _numberOfParentsToRetain = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setNumberOfMutationsPerGeneration(value: Int): this.type = {
    _numberOfMutationsPerGeneration = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setGeneticMixing(value: Double): this.type = {
    _geneticMixing = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setGenerationalMutationStrategy(value: String): this.type = {
    _generationalMutationStrategy = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setFixedMutationValue(value: Int): this.type = {
    _fixedMutationValue = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setMutationMagnitudeMode(value: String): this.type = {
    _mutationMagnitudeMode = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setMlFlowConfig(value: MLFlowConfig): this.type = {
    _mlFlowConfig = value
    setConfigs()
    this
  }

  def mlFlowLoggingOn(): this.type = {
    _mlFlowLoggingFlag = true
    setConfigs()
    this
  }

  def mlFlowLoggingOff(): this.type = {
    _mlFlowLoggingFlag = false
    setConfigs()
    this
  }

  def setMlFlowTrackingURI(value: String): this.type = {
    _mlFlowTrackingURI = value
    setMlFlowConfig()
    setConfigs()
    this
  }

  def setMlFlowExperimentName(value: String): this.type = {
    _mlFlowExperimentName = value
    setMlFlowConfig()
    setConfigs()
    this
  }

  def setMlFlowAPIToken(value: String): this.type = {
    _mlFlowAPIToken = value
    setMlFlowConfig()
    setConfigs()
    this
  }

  def setMlFlowModelSaveDirectory(value: String): this.type = {
    _mlFlowModelSaveDirectory = value
    setMlFlowConfig()
    setConfigs()
    this
  }

  private def setMlFlowConfig(): this.type = {
    _mlFlowConfig = MLFlowConfig(
      mlFlowTrackingURI = _mlFlowTrackingURI,
      mlFlowExperimentName = _mlFlowExperimentName,
      mlFlowAPIToken = _mlFlowTrackingURI,
      mlFlowModelSaveDirectory = _mlFlowModelSaveDirectory
    )
    this
  }

  def autoStoppingOn(): this.type = {
    _autoStoppingFlag = true
    setConfigs()
    this
  }

  def autoStoppingOff(): this.type = {
    _autoStoppingFlag = false
    setConfigs()
    this
  }

  def setAutoStoppingScore(value: Double): this.type = {
    _autoStoppingScore = value
    setConfigs()
    this
  }

  def setFeatureImportanceCutoffType(value: String): this.type = {

    require(_supportedFeatureImportanceCutoffTypes.contains(value),
      s"Feature Importance Cutoff Type '$value' is not supported.  Allowable values: " +
        s"${_supportedFeatureImportanceCutoffTypes.mkString(" ,")}")
    _featureImportanceCutoffType = value
    setConfigs()
    this
  }

  def setFeatureImportanceCutoffValue(value: Double): this.type = {
    _featureImportanceCutoffValue = value
    setConfigs()
    this
  }

  def setEvolutionStrategy(value: String): this.type = {
    require(_allowableEvolutionStrategies.contains(value),
      s"Evolution Strategy '$value' is not a supported mode.  Must be one of: ${
        _allowableEvolutionStrategies.mkString(", ")
      }")
    _evolutionStrategy = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setContinuousEvolutionMaxIterations(value: Int): this.type = {
    if (value > 500) println(s"[WARNING] Total Modeling count $value is higher than recommended limit of 500.  " +
      s"This tuning will take a long time to run.")
    _continuousEvolutionMaxIterations = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setContinuousEvolutionStoppingScore(value: Double): this.type = {
    _continuousEvolutionStoppingScore = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setContinuousEvolutionParallelism(value: Int): this.type = {
    if (value > 10) println(s"[WARNING] ContinuousEvolutionParallelism -> $value is higher than recommended " +
      s"concurrency for efficient optimization for convergence." +
      s"\n  Setting this value below 11 will converge faster in most cases.")
    _continuousEvolutionParallelism = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setContinuousEvolutionMutationAggressiveness(value: Int): this.type = {
    if (value > 4) println(s"[WARNING] ContinuousEvolutionMutationAggressiveness -> $value. " +
      s"\n  Setting this higher than 4 will result in extensive random search and will take longer to converge " +
      s"to optimal hyperparameters.")
    _continuousEvolutionMutationAggressiveness = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setContinuousEvolutionGeneticMixing(value: Double): this.type = {
    require(value < 1.0 & value > 0.0,
      s"Mutation Aggressiveness must be in range (0,1). Current Setting of $value is not permitted.")
    _continuousEvolutionGeneticMixing = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setContinuousEvolutionRollingImprovementCount(value: Int): this.type = {
    require(value > 0, s"ContinuousEvolutionRollingImprovementCount must be > 0. $value is invalid.")
    if (value < 10) println(s"[WARNING] ContinuousEvolutionRollingImprovementCount -> $value setting is low.  " +
      s"Optimal Convergence may not occur due to early stopping.")
    _continuousEvolutionRollingImprovementCount = value
    setGeneticConfig()
    setConfigs()
    this
  }

  private def setGeneticConfig(): this.type = {
    _geneticConfig = GeneticConfig(
      parallelism = _parallelism,
      kFold = _kFold,
      trainPortion = _trainPortion,
      trainSplitMethod = _trainSplitMethod,
      trainSplitChronologicalColumn = _trainSplitChronologicalColumn,
      trainSplitChronologicalRandomPercentage = _trainSplitChronologicalRandomPercentage,
      seed = _seed,
      firstGenerationGenePool = _firstGenerationGenePool,
      numberOfGenerations = _numberOfGenerations,
      numberOfParentsToRetain = _numberOfParentsToRetain,
      numberOfMutationsPerGeneration = _numberOfMutationsPerGeneration,
      geneticMixing = _geneticMixing,
      generationalMutationStrategy = _generationalMutationStrategy,
      fixedMutationValue = _fixedMutationValue,
      mutationMagnitudeMode = _mutationMagnitudeMode,
      evolutionStrategy = _evolutionStrategy,
      continuousEvolutionMaxIterations = _continuousEvolutionMaxIterations,
      continuousEvolutionStoppingScore = _continuousEvolutionStoppingScore,
      continuousEvolutionParallelism = _continuousEvolutionParallelism,
      continuousEvolutionMutationAggressiveness = _continuousEvolutionMutationAggressiveness,
      continuousEvolutionGeneticMixing = _continuousEvolutionGeneticMixing,
      continuousEvolutionRollingImprovementCount = _continuousEvolutionRollingImprovementCount
    )
    this
  }

  def setMainConfig(): this.type = {
    _mainConfig = MainConfig(
      modelFamily = _modelingFamily,
      labelCol = _labelCol,
      featuresCol = _featuresCol,
      naFillFlag = _naFillFlag,
      varianceFilterFlag = _varianceFilterFlag,
      outlierFilterFlag = _outlierFilterFlag,
      pearsonFilteringFlag = _pearsonFilterFlag,
      covarianceFilteringFlag = _covarianceFilterFlag,
      scalingFlag = _scalingFlag,
      autoStoppingFlag = _autoStoppingFlag,
      autoStoppingScore = _autoStoppingScore,
      featureImportanceCutoffType = _featureImportanceCutoffType,
      featureImportanceCutoffValue = _featureImportanceCutoffValue,
      dateTimeConversionType = _dateTimeConversionType,
      fieldsToIgnoreInVector = _fieldsToIgnoreInVector,
      numericBoundaries = _numericBoundaries,
      stringBoundaries = _stringBoundaries,
      scoringMetric = _scoringMetric,
      scoringOptimizationStrategy = _scoringOptimizationStrategy,
      fillConfig = _fillConfig,
      outlierConfig = _outlierConfig,
      pearsonConfig = _pearsonConfig,
      covarianceConfig = _covarianceConfig,
      scalingConfig = _scalingConfig,
      geneticConfig = _geneticConfig,
      mlFlowLoggingFlag = _mlFlowLoggingFlag,
      mlFlowConfig = _mlFlowConfig
    )
    this
  }

  def setMainConfig(value: MainConfig): this.type = {
    _mainConfig = value
    this
  }

  def setFeatConfig(): this.type = {
    _featureImportancesConfig = MainConfig(
      modelFamily = "RandomForest",
      labelCol = _labelCol,
      featuresCol = _featuresCol,
      naFillFlag = _naFillFlag,
      varianceFilterFlag = _varianceFilterFlag,
      outlierFilterFlag = _outlierFilterFlag,
      pearsonFilteringFlag = _pearsonFilterFlag,
      covarianceFilteringFlag = _covarianceFilterFlag,
      scalingFlag = _scalingFlag,
      autoStoppingFlag = _autoStoppingFlag,
      autoStoppingScore = _autoStoppingScore,
      featureImportanceCutoffType = _featureImportanceCutoffType,
      featureImportanceCutoffValue = _featureImportanceCutoffValue,
      dateTimeConversionType = _dateTimeConversionType,
      fieldsToIgnoreInVector = _fieldsToIgnoreInVector,
      numericBoundaries = _numericBoundaries,
      stringBoundaries = _stringBoundaries,
      scoringMetric = _scoringMetric,
      scoringOptimizationStrategy = _scoringOptimizationStrategy,
      fillConfig = _fillConfig,
      outlierConfig = _outlierConfig,
      pearsonConfig = _pearsonConfig,
      covarianceConfig = _covarianceConfig,
      scalingConfig = _scalingConfig,
      geneticConfig = _geneticConfig,
      mlFlowLoggingFlag = _mlFlowLoggingFlag,
      mlFlowConfig = _mlFlowConfig
    )
    this
  }

  def setFeatConfig(value: MainConfig): this.type = {
    _featureImportancesConfig = value
    require(value.modelFamily == "RandomForest",
      s"Model Family for Feature Importances must be 'RandomForest'. ${value.modelFamily} is not supported.")
    setConfigs()
    this
  }

  def setTreeSplitsConfig(): this.type = {
    _treeSplitsConfig = MainConfig(
      modelFamily = "Trees",
      labelCol = _labelCol,
      featuresCol = _featuresCol,
      naFillFlag = _naFillFlag,
      varianceFilterFlag = _varianceFilterFlag,
      outlierFilterFlag = _outlierFilterFlag,
      pearsonFilteringFlag = _pearsonFilterFlag,
      covarianceFilteringFlag = _covarianceFilterFlag,
      scalingFlag = _scalingFlag,
      autoStoppingFlag = _autoStoppingFlag,
      autoStoppingScore = _autoStoppingScore,
      featureImportanceCutoffType = _featureImportanceCutoffType,
      featureImportanceCutoffValue = _featureImportanceCutoffValue,
      dateTimeConversionType = _dateTimeConversionType,
      fieldsToIgnoreInVector = _fieldsToIgnoreInVector,
      numericBoundaries = _numericBoundaries,
      stringBoundaries = _stringBoundaries,
      scoringMetric = _scoringMetric,
      scoringOptimizationStrategy = _scoringOptimizationStrategy,
      fillConfig = _fillConfig,
      outlierConfig = _outlierConfig,
      pearsonConfig = _pearsonConfig,
      covarianceConfig = _covarianceConfig,
      scalingConfig = _scalingConfig,
      geneticConfig = _geneticConfig,
      mlFlowLoggingFlag = _mlFlowLoggingFlag,
      mlFlowConfig = _mlFlowConfig
    )
    this
  }

  def setTreeSplitsConfig(value: MainConfig): this.type = {
    _treeSplitsConfig = value
    require(value.modelFamily == "Trees",
      s"Model Family for Trees Splits must be 'Trees'. ${value.modelFamily} is not supported.")
    setConfigs()
    this
  }

  def getModelingFamily: String = _modelingFamily

  def getLabelCol: String = _labelCol

  def getFeaturesCol: String = _featuresCol

  def getNaFillStatus: Boolean = _naFillFlag

  def getVarianceFilterStatus: Boolean = _varianceFilterFlag

  def getOutlierFilterStatus: Boolean = _outlierFilterFlag

  def getPearsonFilterStatus: Boolean = _pearsonFilterFlag

  def getCovarianceFilterStatus: Boolean = _covarianceFilterFlag

  def getNumericBoundaries: Map[String, (Double, Double)] = _numericBoundaries

  def getStringBoundaries: Map[String, List[String]] = _stringBoundaries

  def getNumericFillStat: String = _numericFillStat

  def getCharacterFillStat: String = _characterFillStat

  def getDateTimeConversionType: String = _dateTimeConversionType

  def getFieldsToIgnoreInVector: Array[String] = _fieldsToIgnoreInVector

  def getModelSelectionDistinctThreshold: Int = _modelSelectionDistinctThreshold

  def getFillConfig: FillConfig = _fillConfig

  def getFilterBounds: String = _filterBounds

  def getLowerFilterNTile: Double = _lowerFilterNTile

  def getUpperFilterNTile: Double = _upperFilterNTile

  def getFilterPrecision: Double = _filterPrecision

  def getContinuousDataThreshold: Int = _continuousDataThreshold

  def getFieldsToIgnore: Array[String] = _fieldsToIgnore

  def getOutlierConfig: OutlierConfig = _outlierConfig

  def getPearsonFilterStatistic: String = _pearsonFilterStatistic

  def getPearsonFilterDirection: String = _pearsonFilterDirection

  def getPearsonFilterManualValue: Double = _pearsonFilterManualValue

  def getPearsonFilterMode: String = _pearsonFilterMode

  def getPearsonAutoFilterNTile: Double = _pearsonAutoFilterNTile

  def getPearsonConfig: PearsonConfig = _pearsonConfig

  def getCorrelationCutoffLow: Double = _correlationCutoffLow

  def getCorrelationCutoffHigh: Double = _correlationCutoffHigh

  def getCovarianceConfig: CovarianceConfig = _covarianceConfig

  def getScalerType: String = _scalerType

  def getScalerMin: Double = _scalerMin

  def getScalerMax: Double = _scalerMax

  def getStandardScalingMeanFlag: Boolean = _standardScalerMeanFlag

  def getStandardScalingStdDevFlag: Boolean = _standardScalerStdDevFlag

  def getPNorm: Double = _pNorm

  def getScalingConfig: ScalingConfig = _scalingConfig

  def getParallelism: Int = _parallelism

  def getKFold: Int = _kFold

  def getTrainPortion: Double = _trainPortion

  def getTrainSplitMethod: String = _trainSplitMethod

  def getTrainSplitChronologicalColumn: String = _trainSplitChronologicalColumn

  def getTrainSplitChronologicalRandomPercentage: Double = _trainSplitChronologicalRandomPercentage

  def getSeed: Long = _seed

  def getFirstGenerationGenePool: Int = _firstGenerationGenePool

  def getNumberOfGenerations: Int = _numberOfGenerations

  def getNumberOfParentsToRetain: Int = _numberOfParentsToRetain

  def getNumberOfMutationsPerGeneration: Int = _numberOfMutationsPerGeneration

  def getGeneticMixing: Double = _geneticMixing

  def getGenerationalMutationStrategy: String = _generationalMutationStrategy

  def getFixedMutationValue: Int = _fixedMutationValue

  def getMutationMagnitudeMode: String = _mutationMagnitudeMode

  def getMlFlowLoggingFlag: Boolean = _mlFlowLoggingFlag

  def getMlFlowTrackingURI: String = _mlFlowTrackingURI

  def getMlFlowExperimentName: String = _mlFlowExperimentName

  def getMlFlowModelSaveDirectory: String = _mlFlowModelSaveDirectory

  def getGeneticConfig: GeneticConfig = _geneticConfig

  def getMainConfig: MainConfig = _mainConfig

  def getFeatConfig: MainConfig = _featureImportancesConfig

  def getTreeSplitsConfig: MainConfig = _treeSplitsConfig

  def getAutoStoppingFlag: Boolean = _autoStoppingFlag

  def getAutoStoppingScore: Double = _autoStoppingScore

  def getFeatureImportanceCutoffType: String = _featureImportanceCutoffType

  def getFeatureImportanceCutoffValue: Double = _featureImportanceCutoffValue

  def getEvolutionStrategy: String = _evolutionStrategy

  def getContinuousEvolutionMaxIterations: Int = _continuousEvolutionMaxIterations

  def getContinuousEvolutionStoppingScore: Double = _continuousEvolutionStoppingScore

  def getContinuousEvolutionParallelism: Int = _continuousEvolutionParallelism

  def getContinuousEvolutionMutationAggressiveness: Int = _continuousEvolutionMutationAggressiveness

  def getContinuousEvolutionGeneticMixing: Double = _continuousEvolutionGeneticMixing

  def getContinuousEvolutionRollingImporvementCount: Int = _continuousEvolutionRollingImprovementCount
}
