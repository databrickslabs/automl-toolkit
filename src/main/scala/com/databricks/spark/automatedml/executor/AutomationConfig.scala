package com.databricks.spark.automatedml.executor

import com.databricks.spark.automatedml.params._

trait AutomationConfig extends Defaults {


  var _modelingFamily: String = _defaultModelingFamily

  var _labelCol: String = _defaultLabelCol

  var _featuresCol: String = _defaultFeaturesCol

  var _naFillFlag: Boolean = _defaultNAFillFlag

  var _varianceFilterFlag: Boolean = _defaultVarianceFilterFlag

  var _outlierFilterFlag: Boolean = _defaultOutlierFilterFlag

  var _pearsonFilterFlag: Boolean = _defaultPearsonFilterFlag

  var _covarianceFilterFlag: Boolean = _defaultCovarianceFilterFlag

  var _numericBoundaries: Map[String, (Double, Double)] = _rfDefaultNumBoundaries

  var _stringBoundaries: Map[String, List[String]] = _rfDefaultStringBoundaries

  var _scoringMetric: String = _scoringDefaultClassifier

  var _scoringOptimizationStrategy: String = _scoringOptimizationStrategyClassifier

  var _numericFillStat: String = _fillConfigDefaults.numericFillStat

  var _characterFillStat: String = _fillConfigDefaults.characterFillStat

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

  var _parallelism: Int = _geneticTunerDefaults.parallelism

  var _kFold: Int = _geneticTunerDefaults.kFold

  var _trainPortion: Double = _geneticTunerDefaults.trainPortion

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
    setMainConfig()
    this
  }

  def setLabelCol(value: String): this.type = {
    _labelCol = value
    setMainConfig()
    this
  }

  def setFeaturesCol(value: String): this.type = {
    _featuresCol = value
    setMainConfig()
    this
  }

  def naFillOn(): this.type = {
    _naFillFlag = true
    setMainConfig()
    this
  }

  def naFillOff(): this.type = {
    _naFillFlag = false
    setMainConfig()
    this
  }

  def varianceFilterOn(): this.type = {
    _varianceFilterFlag = true
    setMainConfig()
    this
  }

  def varianceFilterOff(): this.type = {
    _varianceFilterFlag = false
    setMainConfig()
    this
  }

  def outlierFilterOn(): this.type = {
    _outlierFilterFlag = true
    setMainConfig()
    this
  }

  def outlierFilterOff(): this.type = {
    _outlierFilterFlag = false
    setMainConfig()
    this
  }

  def pearsonFilterOn(): this.type = {
    _pearsonFilterFlag = true
    setMainConfig()
    this
  }

  def pearsonFilterOff(): this.type = {
    _pearsonFilterFlag = false
    setMainConfig()
    this
  }

  def covarianceFilterOn(): this.type = {
    _covarianceFilterFlag = true
    setMainConfig()
    this
  }

  def covarianceFilterOff(): this.type = {
    _covarianceFilterFlag = false
    setMainConfig()
    this
  }

  def setNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    _numericBoundaries = value
    setMainConfig()
    this
  }

  def setStringBoundaries(value: Map[String, List[String]]): this.type = {
    _stringBoundaries = value
    setMainConfig()
    this
  }

  def setNumericFillStat(value: String): this.type = {
    _numericFillStat = value
    setFillConfig()
    setMainConfig()
    this
  }

  def setCharacterFillStat(value: String): this.type = {
    _characterFillStat = value
    setFillConfig()
    setMainConfig()
    this
  }

  def setModelSelectionDistinctThreshold(value: Int): this.type = {
    _modelSelectionDistinctThreshold = value
    setFillConfig()
    setMainConfig()
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
    setMainConfig()
    this
  }

  def setLowerFilterNTile(value: Double): this.type = {
    _lowerFilterNTile = value
    setOutlierConfig()
    setMainConfig()
    this
  }

  def setUpperFilterNTile(value: Double): this.type = {
    _upperFilterNTile = value
    setOutlierConfig()
    setMainConfig()
    this
  }

  def setFilterPrecision(value: Double): this.type = {
    _filterPrecision = value
    setOutlierConfig()
    setMainConfig()
    this
  }

  def setContinuousDataThreshold(value: Int): this.type = {
    _continuousDataThreshold = value
    setOutlierConfig()
    setMainConfig()
    this
  }

  def setFieldsToIgnore(value: Array[String]): this.type = {
    _fieldsToIgnore = value
    setOutlierConfig()
    setMainConfig()
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
    setMainConfig()
    this
  }

  def setPearsonFilterDirection(value: String): this.type = {
    _pearsonFilterDirection = value
    setPearsonConfig()
    setMainConfig()
    this
  }

  def setPearsonFilterManualValue(value: Double): this.type = {
    _pearsonFilterManualValue = value
    setPearsonConfig()
    setMainConfig()
    this
  }

  def setPearsonFilterMode(value: String): this.type = {
    _pearsonFilterMode = value
    setPearsonConfig()
    setMainConfig()
    this
  }

  def setPearsonAutoFilterNTile(value: Double): this.type = {
    _pearsonAutoFilterNTile = value
    setPearsonConfig()
    setMainConfig()
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
    setMainConfig()
    this
  }

  def setCorrelationCutoffHigh(value: Double): this.type = {
    _correlationCutoffHigh = value
    setCovarianceConfig()
    setMainConfig()
    this
  }

  private def setCovarianceConfig(): this.type = {
    _covarianceConfig = CovarianceConfig(
      correlationCutoffLow = _correlationCutoffLow,
      correlationCutoffHigh = _correlationCutoffHigh
    )
    this
  }

  def setParallelism(value: Integer): this.type = {
    //TODO: FIND OUT WHAT THIS RESTRICTION NEEDS TO BE FOR PARALLELISM.
    require(_parallelism < 10000, s"Parallelism above 10000 will result in cluster instability.")
    _parallelism = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  def setKFold(value: Integer): this.type = {
    _kFold = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  def setTrainPortion(value: Double): this.type = {
    _trainPortion = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  def setSeed(value: Long): this.type = {
    _seed = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  def setFirstGenerationGenePool(value: Int): this.type = {
    _firstGenerationGenePool = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  def setNumberOfGenerations(value: Int): this.type = {
    _numberOfGenerations = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  def setNumberOfParentsToRetain(value: Int): this.type = {
    _numberOfParentsToRetain = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  def setNumberOfMutationsPerGeneration(value: Int): this.type = {
    _numberOfMutationsPerGeneration = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  def setGeneticMixing(value: Double): this.type = {
    _geneticMixing = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  def setGenerationalMutationStrategy(value: String): this.type = {
    _generationalMutationStrategy = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  def setFixedMutationValue(value: Int): this.type = {
    _fixedMutationValue = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  def setMutationMagnitudeMode(value: String): this.type = {
    _mutationMagnitudeMode = value
    setGeneticConfig()
    setMainConfig()
    this
  }

  private def setGeneticConfig(): this.type = {
    _geneticConfig = GeneticConfig(
      parallelism = _parallelism,
      kFold = _kFold,
      trainPortion = _trainPortion,
      seed = _seed,
      firstGenerationGenePool = _firstGenerationGenePool,
      numberOfGenerations = _numberOfGenerations,
      numberOfParentsToRetain = _numberOfParentsToRetain,
      numberOfMutationsPerGeneration = _numberOfMutationsPerGeneration,
      geneticMixing = _geneticMixing,
      generationalMutationStrategy = _generationalMutationStrategy,
      fixedMutationValue = _fixedMutationValue,
      mutationMagnitudeMode = _mutationMagnitudeMode
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
      numericBoundaries = _numericBoundaries,
      stringBoundaries = _stringBoundaries,
      scoringMetric = _scoringMetric,
      scoringOptimizationStrategy = _scoringOptimizationStrategy,
      fillConfig = _fillConfig,
      outlierConfig = _outlierConfig,
      pearsonConfig = _pearsonConfig,
      covarianceConfig = _covarianceConfig,
      geneticConfig = _geneticConfig
    )
    this
  }

  def setMainConfig(value: MainConfig): this.type = {
    _mainConfig = value
    this
  }

  def setFeatConfig(value: MainConfig): this.type = {
    _featureImportancesConfig = value
    this
  }

  def setTreeSplitsConfig(value: MainConfig): this.type = {
    _treeSplitsConfig = value
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

  def getParallelism: Int = _parallelism

  def getKFold: Int = _kFold

  def getTrainPortion: Double = _trainPortion

  def getSeed: Long = _seed

  def getFirstGenerationGenePool: Int = _firstGenerationGenePool

  def getNumberOfGenerations: Int = _numberOfGenerations

  def getNumberOfParentsToRetain: Int = _numberOfParentsToRetain

  def getNumberOfMutationsPerGeneration: Int = _numberOfMutationsPerGeneration

  def getGeneticMixing: Double = _geneticMixing

  def getGenerationalMutationStrategy: String = _generationalMutationStrategy

  def getFixedMutationValue: Int = _fixedMutationValue

  def getMutationMagnitudeMode: String = _mutationMagnitudeMode

  def getGeneticConfig: GeneticConfig = _geneticConfig

  def getMainConfig: MainConfig = _mainConfig

  def getFeatConfig: MainConfig = _featureImportancesConfig

  def getTreeSplitsConfig: MainConfig = _treeSplitsConfig
}
