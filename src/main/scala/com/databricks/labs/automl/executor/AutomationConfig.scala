package com.databricks.labs.automl.executor

import com.databricks.labs.automl.params._
import com.databricks.labs.automl.sanitize.SanitizerDefaults

trait AutomationConfig extends Defaults with SanitizerDefaults {

  var _modelingFamily: String = _defaultModelingFamily

  var _labelCol: String = _defaultLabelCol

  var _featuresCol: String = _defaultFeaturesCol

  var _naFillFlag: Boolean = _defaultNAFillFlag

  var _varianceFilterFlag: Boolean = _defaultVarianceFilterFlag

  var _outlierFilterFlag: Boolean = _defaultOutlierFilterFlag

  var _pearsonFilterFlag: Boolean = _defaultPearsonFilterFlag

  var _covarianceFilterFlag: Boolean = _defaultCovarianceFilterFlag

  var _oneHotEncodeFlag: Boolean = _defaultOneHotEncodeFlag

  var _scalingFlag: Boolean = _defaultScalingFlag

  var _featureInteractionFlag: Boolean = _defaultFeatureInteractionFlag

  var _dataPrepCachingFlag: Boolean = _defaultDataPrepCachingFlag

  var _dataPrepParallelism: Int = _defaultDataPrepParallelism

  var _numericBoundaries: Map[String, (Double, Double)] =
    _rfDefaultNumBoundaries

  var _stringBoundaries: Map[String, List[String]] = _rfDefaultStringBoundaries

  var _scoringMetric: String = _scoringDefaultClassifier

  var _scoringOptimizationStrategy: String =
    _scoringOptimizationStrategyClassifier

  var _numericFillStat: String = _fillConfigDefaults.numericFillStat

  var _characterFillStat: String = _fillConfigDefaults.characterFillStat

  var _dateTimeConversionType: String = _defaultDateTimeConversionType

  var _fieldsToIgnoreInVector: Array[String] = _defaultFieldsToIgnoreInVector

  var _naFillFilterPrecision: Double = _fillConfigDefaults.filterPrecision

  var _categoricalNAFillMap: Map[String, String] =
    _fillConfigDefaults.categoricalNAFillMap

  var _numericNAFillMap: Map[String, AnyVal] =
    _fillConfigDefaults.numericNAFillMap

  var _characterNABlanketFillValue: String =
    _fillConfigDefaults.characterNABlanketFillValue

  var _numericNABlanketFillValue: Double =
    _fillConfigDefaults.numericNABlanketFillValue

  var _naFillMode: String = _fillConfigDefaults.naFillMode

  var _cardinalitySwitchFlag: Boolean = _fillConfigDefaults.cardinalitySwitch

  var _cardinalityType: String = _fillConfigDefaults.cardinalityType

  var _cardinalityLimit: Int = _fillConfigDefaults.cardinalityLimit

  var _cardinalityPrecision: Double = _fillConfigDefaults.cardinalityPrecision

  var _cardinalityCheckMode: String = _fillConfigDefaults.cardinalityCheckMode

  var _modelSelectionDistinctThreshold: Int =
    _fillConfigDefaults.modelSelectionDistinctThreshold

  var _fillConfig: FillConfig = _fillConfigDefaults

  var _filterBounds: String = _outlierConfigDefaults.filterBounds

  var _lowerFilterNTile: Double = _outlierConfigDefaults.lowerFilterNTile

  var _upperFilterNTile: Double = _outlierConfigDefaults.upperFilterNTile

  var _filterPrecision: Double = _outlierConfigDefaults.filterPrecision

  var _continuousDataThreshold: Int =
    _outlierConfigDefaults.continuousDataThreshold

  var _fieldsToIgnore: Array[String] = _outlierConfigDefaults.fieldsToIgnore

  var _outlierConfig: OutlierConfig = _outlierConfigDefaults

  var _pearsonFilterStatistic: String = _pearsonConfigDefaults.filterStatistic

  var _pearsonFilterDirection: String = _pearsonConfigDefaults.filterDirection

  var _pearsonFilterManualValue: Double =
    _pearsonConfigDefaults.filterManualValue

  var _pearsonFilterMode: String = _pearsonConfigDefaults.filterMode

  var _pearsonAutoFilterNTile: Double = _pearsonConfigDefaults.autoFilterNTile

  var _pearsonConfig: PearsonConfig = _pearsonConfigDefaults

  var _correlationCutoffLow: Double =
    _covarianceConfigDefaults.correlationCutoffLow

  var _correlationCutoffHigh: Double =
    _covarianceConfigDefaults.correlationCutoffHigh

  var _covarianceConfig: CovarianceConfig = _covarianceConfigDefaults

  var _scalerType: String = defaultScalerType

  var _scalerMin: Double = defaultScalerMin

  var _scalerMax: Double = defaultScalerMax

  var _standardScalerMeanFlag: Boolean = defaultStandardScalerMeanFlag

  var _standardScalerStdDevFlag: Boolean = defaultStandardScalerStdDevFlag

  var _pNorm: Double = defaultPNorm

  var _scalingConfig: ScalingConfig = _scalingConfigDefaults

  var _featureInteractionConfig: FeatureInteractionConfig =
    _defaultFeatureInteractionConfig

  var _parallelism: Int = _geneticTunerDefaults.parallelism

  var _kFold: Int = _geneticTunerDefaults.kFold

  var _trainPortion: Double = _geneticTunerDefaults.trainPortion

  var _trainSplitMethod: String = _geneticTunerDefaults.trainSplitMethod

  var _kSampleConfig: KSampleConfig = _geneticTunerDefaults.kSampleConfig

  var _syntheticCol: String = _geneticTunerDefaults.kSampleConfig.syntheticCol

  var _kGroups: Int = _geneticTunerDefaults.kSampleConfig.kGroups

  var _kMeansMaxIter: Int = _geneticTunerDefaults.kSampleConfig.kMeansMaxIter

  var _kMeansTolerance: Double =
    _geneticTunerDefaults.kSampleConfig.kMeansTolerance

  var _kMeansDistanceMeasurement: String =
    _geneticTunerDefaults.kSampleConfig.kMeansDistanceMeasurement

  var _kMeansSeed: Long = _geneticTunerDefaults.kSampleConfig.kMeansSeed

  var _kMeansPredictionCol: String =
    _geneticTunerDefaults.kSampleConfig.kMeansPredictionCol

  var _lshHashTables: Int = _geneticTunerDefaults.kSampleConfig.lshHashTables

  var _lshSeed: Long = _geneticTunerDefaults.kSampleConfig.lshSeed

  var _lshOutputCol: String = _geneticTunerDefaults.kSampleConfig.lshOutputCol

  var _quorumCount: Int = _geneticTunerDefaults.kSampleConfig.quorumCount

  var _minimumVectorCountToMutate: Int =
    _geneticTunerDefaults.kSampleConfig.minimumVectorCountToMutate

  var _vectorMutationMethod: String =
    _geneticTunerDefaults.kSampleConfig.vectorMutationMethod

  var _mutationMode: String = _geneticTunerDefaults.kSampleConfig.mutationMode

  var _mutationValue: Double = _geneticTunerDefaults.kSampleConfig.mutationValue

  var _labelBalanceMode: String =
    _geneticTunerDefaults.kSampleConfig.labelBalanceMode

  var _cardinalityThreshold: Int =
    _geneticTunerDefaults.kSampleConfig.cardinalityThreshold

  var _numericRatio: Double = _geneticTunerDefaults.kSampleConfig.numericRatio

  var _numericTarget: Int = _geneticTunerDefaults.kSampleConfig.numericTarget

  var _outputDfRepartitionScaleFactor: Int =
    _geneticTunerDefaults.kSampleConfig.outputDfRepartitionScaleFactor

  var _trainSplitChronologicalColumn: String =
    _geneticTunerDefaults.trainSplitChronologicalColumn

  var _trainSplitChronologicalRandomPercentage: Double =
    _geneticTunerDefaults.trainSplitChronologicalRandomPercentage

  var _trainSplitColumnSet: Boolean = false

  var _seed: Long = _geneticTunerDefaults.seed

  var _firstGenerationGenePool: Int =
    _geneticTunerDefaults.firstGenerationGenePool

  var _numberOfGenerations: Int = _geneticTunerDefaults.numberOfGenerations

  var _numberOfParentsToRetain: Int =
    _geneticTunerDefaults.numberOfParentsToRetain

  var _numberOfMutationsPerGeneration: Int =
    _geneticTunerDefaults.numberOfMutationsPerGeneration

  var _geneticMixing: Double = _geneticTunerDefaults.geneticMixing

  var _generationalMutationStrategy: String =
    _geneticTunerDefaults.generationalMutationStrategy

  var _fixedMutationValue: Int = _geneticTunerDefaults.fixedMutationValue

  var _mutationMagnitudeMode: String =
    _geneticTunerDefaults.mutationMagnitudeMode

  var _modelSeedMap: Map[String, Any] = Map.empty

  var _modelSeedSetStatus: Boolean = false

  var _firstGenerationConfig: FirstGenerationConfig =
    _defaultFirstGenerationConfig

  var _firstGenerationPermutationCount: Int =
    _geneticTunerDefaults.initialGenerationConfig.permutationCount

  var _firstGenerationIndexMixingMode: String =
    _geneticTunerDefaults.initialGenerationConfig.indexMixingMode

  var _firstGenerationArraySeed: Long =
    _geneticTunerDefaults.initialGenerationConfig.arraySeed

  var _hyperSpaceInference: Boolean = _defaultHyperSpaceInference

  var _hyperSpaceInferenceCount: Int = _defaultHyperSpaceInferenceCount

  var _hyperSpaceModelType: String = _defaultHyperSpaceModelType

  var _hyperSpaceModelCount: Int = _defaultHyperSpaceModelCount

  var _firstGenerationMode: String = _defaultInitialGenerationMode

  var _geneticConfig: GeneticConfig = _geneticTunerDefaults

  var _mainConfig: MainConfig = _mainConfigDefaults

  var _featureImportancesConfig: MainConfig = _featureImportancesDefaults

  var _treeSplitsConfig: MainConfig = _treeSplitDefaults

  var _mlFlowConfig: MLFlowConfig = _mlFlowConfigDefaults

  var _mlFlowLoggingFlag: Boolean = _defaultMlFlowLoggingFlag

  var _mlFlowArtifactsFlag: Boolean = _defaultMlFlowArtifactsFlag

  var _mlFlowTrackingURI: String = _mlFlowConfigDefaults.mlFlowTrackingURI

  var _mlFlowExperimentName: String = _mlFlowConfigDefaults.mlFlowExperimentName

  var _mlFlowAPIToken: String = _mlFlowConfigDefaults.mlFlowAPIToken

  var _mlFlowModelSaveDirectory: String =
    _mlFlowConfigDefaults.mlFlowModelSaveDirectory

  var _mlFlowLoggingMode: String = _mlFlowConfigDefaults.mlFlowLoggingMode

  var _mlFlowBestSuffix: String = _mlFlowConfigDefaults.mlFlowBestSuffix

  var _mlFlowCustomRunTags: Map[String, String] =
    _mlFlowConfigDefaults.mlFlowCustomRunTags

  var _autoStoppingFlag: Boolean = _defaultAutoStoppingFlag

  var _autoStoppingScore: Double = _defaultAutoStoppingScore

  var _featureImportanceCutoffType: String = _defaultFeatureImportanceCutoffType

  var _featureImportanceCutoffValue: Double =
    _defaultFeatureImportanceCutoffValue

  var _evolutionStrategy: String = _geneticTunerDefaults.evolutionStrategy

  var _continuousEvolutionImprovementThreshold: Int =
    _geneticTunerDefaults.continuousEvolutionImprovementThreshold

  var _geneticMBORegressorType: String =
    _geneticTunerDefaults.geneticMBORegressorType

  var _geneticMBOCandidateFactor: Int =
    _geneticTunerDefaults.geneticMBOCandidateFactor

  var _continuousEvolutionMaxIterations: Int =
    _geneticTunerDefaults.continuousEvolutionMaxIterations

  var _continuousEvolutionStoppingScore: Double =
    _geneticTunerDefaults.continuousEvolutionStoppingScore

  var _continuousEvolutionParallelism: Int =
    _geneticTunerDefaults.continuousEvolutionParallelism

  var _continuousEvolutionMutationAggressiveness: Int =
    _geneticTunerDefaults.continuousEvolutionMutationAggressiveness

  var _continuousEvolutionGeneticMixing: Double =
    _geneticTunerDefaults.continuousEvolutionGeneticMixing

  var _continuousEvolutionRollingImprovementCount: Int =
    _geneticTunerDefaults.continuousEvolutionRollingImprovementCount

  var _inferenceConfigSaveLocation: String = _inferenceConfigSaveLocationDefault

  var _dataReductionFactor: Double = _defaultDataReductionFactor

  var _pipelineDebugFlag: Boolean = _defaultPipelineDebugFlag

  var _featureInteractionRetentionMode: String =
    _defaultFeatureInteractionConfig.retentionMode
  var _featureInteractionContinuousDiscretizerBucketCount: Int =
    _defaultFeatureInteractionConfig.continuousDiscretizerBucketCount
  var _featureInteractionParallelism: Int =
    _defaultFeatureInteractionConfig.parallelism
  var _featureInteractionTargetInteractionPercentage: Double =
    _defaultFeatureInteractionConfig.targetInteractionPercentage

  private def setConfigs(): this.type = {
    setMainConfig()
  }

  def setModelingFamily(value: String): this.type = {
    _modelingFamily = value
    _numericBoundaries = value match {
      case "RandomForest"       => _rfDefaultNumBoundaries
      case "MLPC"               => _mlpcDefaultNumBoundaries
      case "Trees"              => _treesDefaultNumBoundaries
      case "GBT"                => _gbtDefaultNumBoundaries
      case "LinearRegression"   => _linearRegressionDefaultNumBoundaries
      case "LogisticRegression" => _logisticRegressionDefaultNumBoundaries
      case "SVM"                => _svmDefaultNumBoundaries
      case "XGBoost"            => _xgboostDefaultNumBoundaries
      case "gbmBinary" | "gbmMulti" | "gbmMultiOVA" | "gbmHuber" | "gbmFair" |
          "gbmLasso" | "gbmRidge" | "gbmPoisson" | "gbmQuantile" | "gbmMape" |
          "gbmTweedie" | "gbmGamma" =>
        _lightGBMDefaultNumBoundaries
      case _ =>
        throw new IllegalArgumentException(
          s"$value is an unsupported Model Type"
        )
    }
    _stringBoundaries = value match {
      case "RandomForest"       => _rfDefaultStringBoundaries
      case "MLPC"               => _mlpcDefaultStringBoundaries
      case "Trees"              => _treesDefaultStringBoundaries
      case "GBT"                => _gbtDefaultStringBoundaries
      case "LinearRegression"   => _linearRegressionDefaultStringBoundaries
      case "LogisticRegression" => _logisticRegressionDefaultStringBoundaries
      case "SVM"                => _svmDefaultStringBoundaries
      case "XGBoost"            => Map()
      case "gbmBinary" | "gbmMulti" | "gbmMultiOVA" | "gbmHuber" | "gbmFair" |
          "gbmLasso" | "gbmRidge" | "gbmPoisson" | "gbmQuantile" | "gbmMape" |
          "gbmTweedie" | "gbmGamma" =>
        _lightGBMDefaultStringBoundaries
      case _ =>
        throw new IllegalArgumentException(
          s"$value is an unsupported Model Type"
        )
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

  def oneHotEncodingOn(): this.type = {
    _oneHotEncodeFlag = true
    setConfigs()
    this
  }

  def oneHotEncodingOff(): this.type = {
    _oneHotEncodeFlag = false
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

  def dataPrepCachingOn(): this.type = {
    _dataPrepCachingFlag = true
    setConfigs()
    this
  }

  def dataPrepCachingOff(): this.type = {
    _dataPrepCachingFlag = false
    setConfigs()
    this
  }

  def featureInteractionOn(): this.type = {
    _featureInteractionFlag = true
    setConfigs()
    this
  }

  def featureInteractionOff(): this.type = {
    _featureInteractionFlag = false
    setConfigs()
    this
  }

  /**
    * Setter for defining the number of concurrent threads allocated to performing asynchronous data prep tasks within
    * the feature engineering aspect of this application.
    * @param value Int: A value that must be greater than zero.
    * @note This value has an upper limit, depending on driver size, that will restrict the efficacy of the asynchronous
    *       tasks within the pool.  Setting this too high may cause cluster instability.
    * @author Ben Wilson, Databricks
    * @since 0.6.0
    * @throws IllegalArgumentException if a value less than or equal to zero is supplied.
    */
  @throws(classOf[IllegalArgumentException])
  def setDataPrepParallelism(value: Int): this.type = {

    require(value > 0, s"DataPrepParallelism must be greater than zero.")
    _dataPrepParallelism = value
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

  def setScoringMetric(value: String): this.type = {
    _scoringMetric = value
    setConfigs()
    this
  }

  def setScoringOptimizationStrategy(value: String): this.type = {
    require(
      Array("minimize", "maximize").contains(value),
      s"$value is not a member of allowed scoring optimizations: " +
        s"'minimize' or 'maximize'"
    )
    _scoringOptimizationStrategy = value
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

  /**
    * Setter for defining the precision for calculating the model type as per the label column
    *
    * @note setting this value to zero (0) for a large regression problem will incur a long processing time and
    *       an expensive shuffle.
    * @param value Double: Precision accuracy for approximate distinct calculation.
    * @throws java.lang.AssertionError If the value is outside of the allowable range of {0, 1}
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[AssertionError])
  def setNAFillFilterPrecision(value: Double): this.type = {
    require(
      value >= 0,
      s"Filter Precision for NA Fill must be greater than or equal to 0."
    )
    require(
      value <= 1,
      s"Filter Precision for NA Fill must be less than or equal to 1."
    )
    _naFillFilterPrecision = value
    setFillConfig()
    setConfigs()
    this
  }

  /**
    * Setter for providing a map of [Column Name -> String Fill Value] for manual by-column overrides.  Any non-specified
    * fields in this map will utilize the "auto" statistics-based fill paradigm to calculate and fill any NA values
    * in non-numeric columns.
    *
    * @note if naFillMode is specified as using Map Fill modes, this setter or the numeric na fill map MUST be set.
    * @note If fields are specified in here that are not part of the DataFrame's schema, an exception will be thrown.
    * @param value Map[String, String]: Column Name as String -> Fill Value as String
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  def setCategoricalNAFillMap(value: Map[String, String]): this.type = {
    _categoricalNAFillMap = value
    setFillConfig()
    setConfigs()
    this
  }

  /**
    * Setter for providing a map of [Column Name -> AnyVal Fill Value] (must be numeric). Any non-specified
    * fields in this map will utilize the "auto" statistics-based fill paradigm to calculate and fill any NA values
    * in numeric columns.
    *
    * @note if naFillMode is specified as using Map Fill modes, this setter or the categorical na fill map MUST be set.
    * @note If fields are specified in here that are not part of the DataFrame's schema, an exception will be thrown.
    * @param value Map[String, AnyVal]: Column Name as String -> Fill Numeric Type Value
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  def setNumericNAFillMap(value: Map[String, AnyVal]): this.type = {
    _numericNAFillMap = value
    setFillConfig()
    setConfigs()
    this
  }

  /**
    * Setter for providing a 'blanket override' value (fill all found categorical columns' missing values with this
    * specified value).
    *
    * @param value String: A value to fill all categorical na values in the DataFrame with.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  def setCharacterNABlanketFillValue(value: String): this.type = {
    _characterNABlanketFillValue = value
    setFillConfig()
    setConfigs()
    this
  }

  /**
    * Setter for providing a 'blanket override'  value (fill all found numeric columns' missing values with this
    * specified value)
    *
    * @param value Double: A value to fill all numeric na value in the DataFrame with.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  def setNumericNABlanketFillValue(value: Double): this.type = {
    _numericNABlanketFillValue = value
    setFillConfig()
    setConfigs()
    this
  }

  /**
    * Mode for na fill<br>
    *                Available modes: <br>
    *                  <i>auto</i> : Stats-based na fill for fields.  Usage of .setNumericFillStat and
    *                  .setCharacterFillStat will inform the type of statistics that will be used to fill.<br>
    *                  <i>mapFill</i> : Custom by-column overrides to 'blanket fill' na values on a per-column
    *                  basis.  The categorical (string) fields are set via .setCategoricalNAFillMap while the
    *                  numeric fields are set via .setNumericNAFillMap.<br>
    *                  <i>blanketFillAll</i> : Fills all fields based on the values specified by
    *                  .setCharacterNABlanketFillValue and .setNumericNABlanketFillValue.  All NA's for the
    *                  appropriate types will be filled in accordingly throughout all columns.<br>
    *                  <i>blanketFillCharOnly</i> Will use statistics to fill in numeric fields, but will replace
    *                  all categorical character fields na values with a blanket fill value. <br>
    *                  <i>blanketFillNumOnly</i> Will use statistics to fill in character fields, but will replace
    *                  all numeric fields na values with a blanket value.
    *
    * @throws IllegalArgumentException if the mods specified is not supported.
    * @param value String: Mode for NA Fill
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[IllegalArgumentException])
  def setNAFillMode(value: String): this.type = {
    require(
      _allowableNAFillModes.contains(value),
      s"NA Fill Mode '$value' is not a supported mode.  Must be one of:" +
        s"${_allowableNAFillModes.mkString(", ")}"
    )
    _naFillMode = value
    setFillConfig()
    setConfigs()
    this
  }

  def setModelSelectionDistinctThreshold(value: Int): this.type = {
    _modelSelectionDistinctThreshold = value
    setFillConfig()
    setConfigs()
    this
  }

  def cardinalitySwitchOn(): this.type = {
    _cardinalitySwitchFlag = true
    setFillConfig()
    setConfigs()
    this
  }

  def cardinalitySwitchOff(): this.type = {
    _cardinalitySwitchFlag = false
    setFillConfig()
    setConfigs()
    this
  }
  def setCardinalitySwitch(value: Boolean): this.type = {
    _cardinalitySwitchFlag = value
    setFillConfig()
    setConfigs()
    this
  }

  @throws(classOf[AssertionError])
  def setCardinalityType(value: String): this.type = {
    _cardinalityType = value
    assert(
      allowableCardinalilties.contains(value),
      s"Supplied CardinalityType '$value' is not in: " +
        s"${allowableCardinalilties.mkString(", ")}"
    )
    setFillConfig()
    setConfigs()
    this
  }

  @throws(classOf[IllegalArgumentException])
  def setCardinalityLimit(value: Int): this.type = {
    require(value > 0, s"Cardinality limit must be greater than 0")
    _cardinalityLimit = value
    setFillConfig()
    setConfigs()
    this
  }

  @throws(classOf[IllegalArgumentException])
  def setCardinalityPrecision(value: Double): this.type = {
    require(value >= 0.0, s"Precision must be greater than or equal to 0.")
    require(value <= 1.0, s"Precision must be less than or equal to 1.")
    _cardinalityPrecision = value
    setFillConfig()
    setConfigs()
    this
  }

  @throws(classOf[AssertionError])
  def setCardinalityCheckMode(value: String): this.type = {
    assert(
      allowableCategoricalFilterModes.contains(value),
      s"Supplied CardinalityCheckMode $value is not in: ${allowableCategoricalFilterModes.mkString(", ")}"
    )
    _cardinalityCheckMode = value
    setFillConfig()
    setConfigs()
    this
  }

  private def setFillConfig(): this.type = {
    _fillConfig = FillConfig(
      numericFillStat = _numericFillStat,
      characterFillStat = _characterFillStat,
      modelSelectionDistinctThreshold = _modelSelectionDistinctThreshold,
      cardinalitySwitch = _cardinalitySwitchFlag,
      cardinalityType = _cardinalityType,
      cardinalityLimit = _cardinalityLimit,
      cardinalityPrecision = _cardinalityPrecision,
      cardinalityCheckMode = _cardinalityCheckMode,
      filterPrecision = _naFillFilterPrecision,
      categoricalNAFillMap = _categoricalNAFillMap,
      numericNAFillMap = _numericNAFillMap,
      characterNABlanketFillValue = _characterNABlanketFillValue,
      numericNABlanketFillValue = _numericNABlanketFillValue,
      naFillMode = _naFillMode
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

  /**
    * Setter for determining the mode of operation for inclusion of interacted features.
    * Modes are:
    *   - all -> Includes all interactions between all features (after string indexing of categorical values)
    *   - optimistic -> If the Information Gain / Variance, as compared to at least ONE of the parents of the interaction
    *       is above the threshold set by featureInteractionTargetInteractionPercentage
    *         (e.g. if IG of left parent is 0.5 and right parent is 0.9, with threshold set at 10, if the interaction
    *         between these two parents has an IG of 0.42, it would be rejected, but if it was 0.46, it would be kept)
    *   - strict -> the threshold percentage must be met for BOTH parents.
    *         (in the above example, the IG for the interaction would have to be > 0.81 in order to be included in
    *         the feature vector).
    * @param value String -> one of: 'all', 'optimistic', or 'strict'
    * @throws IllegalArgumentException if the specified value submitted is not permitted
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[IllegalArgumentException])
  def setFeatureInteractionRetentionMode(value: String): this.type = {
    require(
      allowableFeatureInteractionModes.contains(value),
      s"FeatureInteractionRetentionMode is invalid.  Must be one of: ${allowableFeatureInteractionModes
        .mkString(", ")}"
    )
    _featureInteractionRetentionMode = value
    setFeatureInteractionConfig()
    setConfigs()
    this
  }

  /**
    * Setter for determining the behavior of continuous feature columns.  In order to calculate Entropy for a continuous
    * variable, the distribution must be converted to nominal values for estimation of per-split information gain.
    * This setting defines how many nominal categorical values to create out of a continuously distributed feature
    * in order to calculate Entropy.
    * @param value Int -> must be greater than 1
    * @throws IllegalArgumentException if the value specified is <= 1
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def setFeatureInteractionContinuousDiscretizerBucketCount(
    value: Int
  ): this.type = {
    require(
      value > 1,
      s"FeatureInteractionContinuousDiscretizerBucketCount must be greater than 1."
    )
    _featureInteractionContinuousDiscretizerBucketCount = value
    setFeatureInteractionConfig()
    setConfigs()
    this
  }

  /**
    * Setter for configuring the concurrent count for scoring of feature interaction candidates.
    * Due to the nature of these operations, the configuration here may need to be set differently to that of
    * the modeling and general feature engineering phases of the toolkit.  This is highly dependent on the row
    * count of the data set being submitted.
    * @param value Int -> must be greater than 0
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    * @throws IllegalArgumentException if the value is < 1
    */
  @throws(classOf[IllegalArgumentException])
  def setFeatureInteractionParallelism(value: Int): this.type = {
    require(
      value >= 1,
      s"FeatureInteractionParallelism must be set to a value >= 1."
    )
    _featureInteractionParallelism = value
    setFeatureInteractionConfig()
    setConfigs()
    this
  }

  /**
    * Setter for establishing the minimum acceptable InformationGain or Variance allowed for an interaction
    * candidate based on comparison to the scores of its parents.
    * @param value Double in range of -inf -> inf
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def setFeatureInteractionTargetInteractionPercentage(
    value: Double
  ): this.type = {
    _featureInteractionTargetInteractionPercentage = value
    setFeatureInteractionConfig()
    setConfigs()
    this
  }

  /**
    * Private setter for establishing the feature interaction configuration
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def setFeatureInteractionConfig(): this.type = {
    _featureInteractionConfig = FeatureInteractionConfig(
      retentionMode = _featureInteractionRetentionMode,
      continuousDiscretizerBucketCount =
        _featureInteractionContinuousDiscretizerBucketCount,
      parallelism = _featureInteractionParallelism,
      targetInteractionPercentage =
        _featureInteractionTargetInteractionPercentage
    )
    this
  }

  def setParallelism(value: Integer): this.type = {
    //TODO: FIND OUT WHAT THIS RESTRICTION NEEDS TO BE FOR PARALLELISM.
    require(
      _parallelism < 10000,
      s"Parallelism above 10000 will result in cluster instability."
    )
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
    require(
      trainSplitMethods.contains(value),
      s"TrainSplitMethod $value must be one of: ${trainSplitMethods.mkString(", ")}"
    )
    _trainSplitMethod = value
    if (value == "chronological")
      println(
        "[WARNING] setTrainSplitMethod() -> Chronological splits is shuffle-intensive and will increase " +
          "runtime significantly.  Only use if necessary for modeling scenario!"
      )
    setGeneticConfig()
    setConfigs()
    this
  }

  def setKSampleConfig(): this.type = {

    _kSampleConfig = KSampleConfig(
      syntheticCol = _syntheticCol,
      kGroups = _kGroups,
      kMeansMaxIter = _kMeansMaxIter,
      kMeansTolerance = _kMeansTolerance,
      kMeansDistanceMeasurement = _kMeansDistanceMeasurement,
      kMeansSeed = _kMeansSeed,
      kMeansPredictionCol = _kMeansPredictionCol,
      lshHashTables = _lshHashTables,
      lshSeed = _lshSeed,
      lshOutputCol = _lshOutputCol,
      quorumCount = _quorumCount,
      minimumVectorCountToMutate = _minimumVectorCountToMutate,
      vectorMutationMethod = _vectorMutationMethod,
      mutationMode = _mutationMode,
      mutationValue = _mutationValue,
      labelBalanceMode = _labelBalanceMode,
      cardinalityThreshold = _cardinalityThreshold,
      numericRatio = _numericRatio,
      numericTarget = _numericTarget,
      outputDfRepartitionScaleFactor = _outputDfRepartitionScaleFactor
    )
    this
  }

  /**
    * Setter - for setting the name of the Synthetic column name
    *
    * @param value String: A column name that is uniquely not part of the main DataFrame
    * @since 0.5.1
    * @author Ben Wilson
    */
  def setSyntheticCol(value: String): this.type = {
    _syntheticCol = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for specifying the number of K-Groups to generate in the KMeans model
    *
    * @param value Int: number of k groups to generate
    * @return this
    */
  def setKGroups(value: Int): this.type = {
    _kGroups = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for specifying the maximum number of iterations for the KMeans model to go through to converge
    *
    * @param value Int: Maximum limit on iterations
    * @return this
    */
  def setKMeansMaxIter(value: Int): this.type = {
    _kMeansMaxIter = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for Setting the tolerance for KMeans (must be >0)
    *
    * @param value The tolerance value setting for KMeans
    * @see reference: [[http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.clustering.KMeans]]
    *      for further details.
    * @return this
    * @throws IllegalArgumentException() if a value less than 0 is entered
    */
  @throws(classOf[IllegalArgumentException])
  def setKMeansTolerance(value: Double): this.type = {
    require(
      value > 0,
      s"KMeans tolerance value ${value.toString} is out of range.  Must be > 0."
    )
    _kMeansTolerance = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for which distance measurement to use to calculate the nearness of vectors to a centroid
    *
    * @param value String: Options -> "euclidean" or "cosine" Default: "euclidean"
    * @return this
    * @throws IllegalArgumentException() if an invalid value is entered
    */
  @throws(classOf[IllegalArgumentException])
  def setKMeansDistanceMeasurement(value: String): this.type = {
    require(
      allowableKMeansDistanceMeasurements.contains(value),
      s"Kmeans Distance Measurement $value is not " +
        s"a valid mode of operation.  Must be one of: ${allowableKMeansDistanceMeasurements.mkString(", ")}"
    )
    _kMeansDistanceMeasurement = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for a KMeans seed for the clustering algorithm
    *
    * @param value Long: Seed value
    * @return this
    */
  def setKMeansSeed(value: Long): this.type = {
    _kMeansSeed = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for the internal KMeans column for cluster membership attribution
    *
    * @param value String: column name for internal algorithm column for group membership
    * @return this
    */
  def setKMeansPredictionCol(value: String): this.type = {
    _kMeansPredictionCol = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for Configuring the number of Hash Tables to use for MinHashLSH
    *
    * @param value Int: Count of hash tables to use
    * @see [[http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.MinHashLSH]]
    *     for more information
    * @return this
    */
  def setLSHHashTables(value: Int): this.type = {
    _lshHashTables = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for Configuring the Seed value for the LSH MinHash model
    *
    * @param value Long: A Seed value
    * @since 0.5.1
    * @author Ben Wilson
    */
  def setLSHSeed(value: Long): this.type = {
    _lshSeed = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for the internal LSH output hash information column
    *
    * @param value String: column name for the internal MinHashLSH Model transformation value
    * @return this
    */
  def setLSHOutputCol(value: String): this.type = {
    _lshOutputCol = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for how many vectors to find in adjacency to the centroid for generation of synthetic data
    *
    * @note the higher the value set here, the higher the variance in synthetic data generation
    * @param value Int: Number of vectors to find nearest each centroid within the class
    * @return this
    */
  def setQuorumCount(value: Int): this.type = {
    _quorumCount = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for minimum threshold for vector indexes to mutate within the feature vector.
    *
    * @note In vectorMutationMethod "fixed" this sets the fixed count of how many vector positions to mutate.
    *       In vectorMutationMethod "random" this sets the lower threshold for 'at least this many indexes will
    *       be mutated'
    * @param value The minimum (or fixed) number of indexes to mutate.
    * @return this
    */
  def setMinimumVectorCountToMutate(value: Int): this.type = {
    _minimumVectorCountToMutate = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for the Vector Mutation Method
    *
    * @note Options:
    *       "fixed" - will use the value of minimumVectorCountToMutate to select random indexes of this number of indexes.
    *       "random" - will use this number as a lower bound on a random selection of indexes between this and the vector length.
    *       "all" - will mutate all of the vectors.
    * @param value String - the mode to use.
    * @return this
    * @throws IllegalArgumentException() if the mode is not supported.
    */
  @throws(classOf[IllegalArgumentException])
  def setVectorMutationMethod(value: String): this.type = {
    require(
      allowableVectorMutationMethods.contains(value),
      s"Vector Mutation Mode $value is not supported.  " +
        s"Must be one of: ${allowableVectorMutationMethods.mkString(", ")} "
    )
    _vectorMutationMethod = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for the Mutation Mode of the feature vector individual values
    *
    * @note Options:
    *       "weighted" - uses weighted averaging to scale the euclidean distance between the centroid vector and mutation candidate vectors
    *       "random" - randomly selects a position on the euclidean vector between the centroid vector and the candidate mutation vectors
    *       "ratio" - uses a ratio between the values of the centroid vector and the mutation vector    *
    * @param value String: the mode to use.
    * @return this
    * @throws IllegalArgumentException() if the mode is not supported.
    */
  @throws(classOf[IllegalArgumentException])
  def setMutationMode(value: String): this.type = {
    require(
      allowableMutationModes.contains(value),
      s"Mutation Mode $value is not a valid mode of operation.  " +
        s"Must be one of: ${allowableMutationModes.mkString(", ")}"
    )
    _mutationMode = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for specifying the mutation magnitude for the modes 'weighted' and 'ratio' in mutationMode
    *
    * @param value Double: value between 0 and 1 for mutation magnitude adjustment.
    * @note the higher this value, the closer to the centroid vector vs. the candidate mutation vector the synthetic row data will be.
    * @return this
    * @throws IllegalArgumentException() if the value specified is outside of the range (0, 1)
    */
  @throws(classOf[IllegalArgumentException])
  def setMutationValue(value: Double): this.type = {
    require(
      value > 0 & value < 1,
      s"Mutation Value must be between 0 and 1. Value $value is not permitted."
    )
    _mutationValue = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter - for determining the label balance approach mode.
    *
    * @note Available modes: <br>
    *         <i>'match'</i>: Will match all smaller class counts to largest class count.  [WARNING] - May significantly increase memory pressure!<br>
    *         <i>'percentage'</i> Will adjust smaller classes to a percentage value of the largest class count.
    *         <i>'target'</i> Will increase smaller class counts to a fixed numeric target of rows.
    * @param value String: one of: 'match', 'percentage' or 'target'
    * @note Default: "percentage"
    * @since 0.5.1
    * @author Ben Wilson
    * @throws UnsupportedOperationException() if the provided mode is not supported.
    */
  @throws(classOf[UnsupportedOperationException])
  def setLabelBalanceMode(value: String): this.type = {
    require(
      allowableLabelBalanceModes.contains(value),
      s"Label Balance Mode $value is not supported." +
        s"Must be one of: ${allowableLabelBalanceModes.mkString(", ")}"
    )
    _labelBalanceMode = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter - for overriding the cardinality threshold exception threshold.  [WARNING] increasing this value on
    * a sufficiently large data set could incur, during runtime, excessive memory and cpu pressure on the cluster.
    *
    * @param value Int: the limit above which an exception will be thrown for a classification problem wherein the
    *              label distinct count is too large to successfully generate synthetic data.
    * @note Default: 20
    * @since 0.5.1
    * @author Ben Wilson
    */
  def setCardinalityThreshold(value: Int): this.type = {
    _cardinalityThreshold = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter - for specifying the percentage ratio for the mode 'percentage' in setLabelBalanceMode()
    *
    * @param value Double: A fractional double in the range of 0.0 to 1.0.
    * @note Setting this value to 1.0 is equivalent to setting the label balance mode to 'match'
    * @note Default: 0.2
    * @since 0.5.1
    * @author Ben Wilson
    * @throws UnsupportedOperationException() if the provided value is outside of the range of 0.0 -> 1.0
    */
  @throws(classOf[UnsupportedOperationException])
  def setNumericRatio(value: Double): this.type = {
    require(
      value <= 1.0 & value > 0.0,
      s"Invalid Numeric Ratio entered!  Must be between 0 and 1." +
        s"${value.toString} is not valid."
    )
    _numericRatio = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter - for specifying the target row count to generate for 'target' mode in setLabelBalanceMode()
    *
    * @param value Int: The desired final number of rows per minority class label
    * @note [WARNING] Setting this value to too high of a number will greatly increase runtime and memory pressure.
    * @since 0.5.1
    * @author Ben Wilson
    */
  def setNumericTarget(value: Int): this.type = {
    _numericTarget = value
    setKSampleConfig()
    setGeneticConfig()
    setConfigs()
    this
  }

  def setTrainSplitChronologicalColumn(value: String): this.type = {
    _trainSplitChronologicalColumn = value
    val ignoredFields: Array[String] = _fieldsToIgnoreInVector ++ Array(value)
    setFieldsToIgnoreInVector(ignoredFields)
    _trainSplitColumnSet = true
    setGeneticConfig()
    setConfigs()
    this
  }

  def setTrainSplitChronologicalRandomPercentage(value: Double): this.type = {
    _trainSplitChronologicalRandomPercentage = value
    if (value > 10)
      println(
        "[WARNING] setTrainSplitChronologicalRandomPercentage() setting this value above 10 " +
          "percent will cause significant per-run train/test skew and variability in row counts during training.  " +
          "Use higher values only if this is desired."
      )
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

  def setModelSeedString(value: String): this.type = {
    _modelSeedMap = extractGenericModelReturnMap(value)
    _modelSeedSetStatus = true
    setGeneticConfig()
    setConfigs()
    this
  }

  def setModelSeedMap(value: Map[String, Any]): this.type = {
    _modelSeedMap = value
    _modelSeedSetStatus = true
    setGeneticConfig()
    setConfigs()
    this
  }

  private def setFirstGenerationConfig(): this.type = {
    _firstGenerationConfig = FirstGenerationConfig(
      permutationCount = _firstGenerationPermutationCount,
      indexMixingMode = _firstGenerationIndexMixingMode,
      arraySeed = _firstGenerationArraySeed
    )
    setGeneticConfig()
    setConfigs()
    this
  }

  def setFirstGenerationPermutationCount(value: Int): this.type = {
    _firstGenerationPermutationCount = value
    setFirstGenerationConfig()
    this
  }

  def setFirstGenerationIndexMixingMode(value: String): this.type = {
    require(
      _allowableInitialGenerationIndexMixingModes.contains(value),
      s"Invalid First Generation Index Mixing " +
        s"Mode: $value .  First Generation Index Mixing Mode must be one of: " +
        s"${_allowableInitialGenerationIndexMixingModes.mkString(", ")}"
    )
    _firstGenerationIndexMixingMode = value
    setFirstGenerationConfig()
    this
  }

  def setFirstGenerationArraySeed(value: Long): this.type = {
    _firstGenerationArraySeed = value
    setFirstGenerationConfig()
    this
  }

  def hyperSpaceInferenceOn(): this.type = {
    _hyperSpaceInference = true
    setGeneticConfig()
    setConfigs()
    this
  }

  def hyperSpaceInferenceOff(): this.type = {
    _hyperSpaceInference = false
    setGeneticConfig()
    setConfigs()
    this
  }

  def setHyperSpaceInferenceCount(value: Int): this.type = {
    if (value > 500000)
      println(
        "WARNING! Setting permutation counts above 500,000 will put stress on the driver."
      )
    if (value > 1000000)
      throw new UnsupportedOperationException(
        s"Setting permutation above 1,000,000 is not supported" +
          s" due to runtime considerations.  $value is too large of a value."
      )
    _hyperSpaceInferenceCount = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setHyperSpaceModelType(value: String): this.type = {
    require(
      Array("RandomForest", "LinearRegression", "XGBoost").contains(value),
      s"Model type $value is not supported for post " +
        s"modeling hyper space optimization!  Please choose either RandomForest or LinearRegression"
    )
    _hyperSpaceModelType = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setHyperSpaceModelCount(value: Int): this.type = {
    if (value > 50)
      println(
        "WARNING! Setting this value above 50 will incur 50 additional models to be built.  Proceed" +
          "only if this is intended."
      )
    _hyperSpaceModelCount = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setFirstGenerationMode(value: String): this.type = {
    require(
      _allowableInitialGenerationModes.contains(value),
      s"Invalid First Generation Mode: $value . " +
        s"First Generation Mode must be one of : ${_allowableInitialGenerationModes.mkString(", ")}"
    )
    _firstGenerationMode = value
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

  def mlFlowLogArtifactsOn(): this.type = {
    _mlFlowArtifactsFlag = true
    setConfigs()
    this
  }

  def mlFlowLogArtifactsOff(): this.type = {
    _mlFlowArtifactsFlag = false
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

  def setMlFlowLoggingMode(value: String): this.type = {
    require(
      _allowableMlFlowLoggingModes.contains(value),
      s"MlFlow logging mode $value is not permitted.  Must be " +
        s"one of: ${_allowableMlFlowLoggingModes.mkString(",")}"
    )
    _mlFlowLoggingMode = value
    setMlFlowConfig()
    setConfigs()
    this
  }

  def setMlFlowBestSuffix(value: String): this.type = {
    _mlFlowBestSuffix = value
    setMlFlowConfig()
    setConfigs()
    this
  }

  def setMlFlowCustomRunTags(value: Map[String, String]): this.type = {
    _mlFlowCustomRunTags = value
    setMlFlowConfig()
    setConfigs()
    this
  }

  private def setMlFlowConfig(): this.type = {
    _mlFlowConfig = MLFlowConfig(
      mlFlowTrackingURI = _mlFlowTrackingURI,
      mlFlowExperimentName = _mlFlowExperimentName,
      mlFlowAPIToken = _mlFlowAPIToken,
      mlFlowModelSaveDirectory = _mlFlowModelSaveDirectory,
      mlFlowLoggingMode = _mlFlowLoggingMode,
      mlFlowBestSuffix = _mlFlowBestSuffix,
      mlFlowCustomRunTags = _mlFlowCustomRunTags
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

  /**
    * Setter for defining the secondary stopping criteria for continuous training mode ( number of consistentlt
    * not-improving runs to terminate the learning algorithm due to diminishing returns.
    * @param value Negative Integer (an improvement to a priori will reset the counter and subsequent non-improvements
    *              will decrement a mutable counter.  If the counter hits this limit specified in value, the continuous
    *              mode algorithm will stop).
    * @author Ben Wilson, Databricks
    * @since 0.6.0
    * @throws IllegalArgumentException if the value is positive.
    */
  @throws(classOf[IllegalArgumentException])
  def setContinuousEvolutionImprovementThreshold(value: Int): this.type = {
    require(
      value < 0,
      s"ContinuousEvolutionImprovementThreshold must be less than zero.  It is " +
        s"recommended to set this value to less than -4."
    )
    _continuousEvolutionImprovementThreshold = value
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for selecting the type of Regressor to use for the within-epoch generation MBO of candidates
    * @param value String - one of "XGBoost", "LinearRegression" or "RandomForest"
    * @author Ben Wilson, Databricks
    * @since 0.6.0
    * @throws IllegalArgumentException if the value is not supported
    */
  @throws(classOf[IllegalArgumentException])
  def setGeneticMBORegressorType(value: String): this.type = {
    require(
      allowableMBORegressorTypes.contains(value),
      s"GeneticRegressorType $value is not a supported Regressor " +
        s"Type.  Must be one of: ${allowableMBORegressorTypes.mkString(", ")}"
    )
    _geneticMBORegressorType = value
    setGeneticConfig()
    setConfigs()
    this
  }

  /**
    * Setter for defining the factor to be applied to the candidate listing of hyperparameters to generate through
    * mutation for each generation other than the initial and post-modeling optimization phases.  The larger this
    * value (default: 10), the more potential space can be searched.  There is not a large performance hit to this,
    * and as such, values in excess of 100 are viable.
    * @param value Int - a factor to multiply the numberOfMutationsPerGeneration by to generate a count of potential
    *              candidates.
    * @author Ben Wilson, Databricks
    * @since 0.6.0
    * @throws IllegalArgumentException if the value is not greater than zero.
    */
  @throws(classOf[IllegalArgumentException])
  def setGeneticMBOCandidateFactor(value: Int): this.type = {
    require(value > 0, s"GeneticMBOCandidateFactor must be greater than zero.")
    _geneticMBOCandidateFactor = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setFeatureImportanceCutoffType(value: String): this.type = {

    require(
      _supportedFeatureImportanceCutoffTypes.contains(value),
      s"Feature Importance Cutoff Type '$value' is not supported.  Allowable values: " +
        s"${_supportedFeatureImportanceCutoffTypes.mkString(" ,")}"
    )
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
    require(
      _allowableEvolutionStrategies.contains(value),
      s"Evolution Strategy '$value' is not a supported mode.  Must be one of: ${_allowableEvolutionStrategies
        .mkString(", ")}"
    )
    _evolutionStrategy = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setContinuousEvolutionMaxIterations(value: Int): this.type = {
    if (value > 500)
      println(
        s"[WARNING] Total Modeling count $value is higher than recommended limit of 500.  " +
          s"This tuning will take a long time to run."
      )
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
    if (value > 10)
      println(
        s"[WARNING] ContinuousEvolutionParallelism -> $value is higher than recommended " +
          s"concurrency for efficient optimization for convergence." +
          s"\n  Setting this value below 11 will converge faster in most cases."
      )
    _continuousEvolutionParallelism = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setContinuousEvolutionMutationAggressiveness(value: Int): this.type = {
    if (value > 4)
      println(
        s"[WARNING] ContinuousEvolutionMutationAggressiveness -> $value. " +
          s"\n  Setting this higher than 4 will result in extensive random search and will take longer to converge " +
          s"to optimal hyperparameters."
      )
    _continuousEvolutionMutationAggressiveness = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setContinuousEvolutionGeneticMixing(value: Double): this.type = {
    require(
      value < 1.0 & value > 0.0,
      s"Mutation Aggressiveness must be in range (0,1). Current Setting of $value is not permitted."
    )
    _continuousEvolutionGeneticMixing = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setContinuousEvolutionRollingImprovementCount(value: Int): this.type = {
    require(
      value > 0,
      s"ContinuousEvolutionRollingImprovementCount must be > 0. $value is invalid."
    )
    if (value < 10)
      println(
        s"[WARNING] ContinuousEvolutionRollingImprovementCount -> $value setting is low.  " +
          s"Optimal Convergence may not occur due to early stopping."
      )
    _continuousEvolutionRollingImprovementCount = value
    setGeneticConfig()
    setConfigs()
    this
  }

  def setInferenceConfigSaveLocation(value: String): this.type = {
    _inferenceConfigSaveLocation = value
    setConfigs()
    this
  }

  def setDataReductionFactor(value: Double): this.type = {
    require(value > 0, s"Data Reduction Factor must be between 0 and 1")
    require(value < 1, s"Data Reduction Factor must be between 0 and 1")
    _dataReductionFactor = value
    setConfigs()
    this
  }

  private def setGeneticConfig(): this.type = {
    _geneticConfig = GeneticConfig(
      parallelism = _parallelism,
      kFold = _kFold,
      trainPortion = _trainPortion,
      trainSplitMethod = _trainSplitMethod,
      kSampleConfig = _kSampleConfig,
      trainSplitChronologicalColumn = _trainSplitChronologicalColumn,
      trainSplitChronologicalRandomPercentage =
        _trainSplitChronologicalRandomPercentage,
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
      geneticMBORegressorType = _geneticMBORegressorType,
      geneticMBOCandidateFactor = _geneticMBOCandidateFactor,
      continuousEvolutionMaxIterations = _continuousEvolutionMaxIterations,
      continuousEvolutionStoppingScore = _continuousEvolutionStoppingScore,
      continuousEvolutionImprovementThreshold =
        _continuousEvolutionImprovementThreshold,
      continuousEvolutionParallelism = _continuousEvolutionParallelism,
      continuousEvolutionMutationAggressiveness =
        _continuousEvolutionMutationAggressiveness,
      continuousEvolutionGeneticMixing = _continuousEvolutionGeneticMixing,
      continuousEvolutionRollingImprovementCount =
        _continuousEvolutionRollingImprovementCount,
      modelSeed = _modelSeedMap,
      hyperSpaceInference = _hyperSpaceInference,
      hyperSpaceInferenceCount = _hyperSpaceInferenceCount,
      hyperSpaceModelType = _hyperSpaceModelType,
      hyperSpaceModelCount = _hyperSpaceModelCount,
      initialGenerationMode = _firstGenerationMode,
      initialGenerationConfig = _firstGenerationConfig
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
      oneHotEncodeFlag = _oneHotEncodeFlag,
      scalingFlag = _scalingFlag,
      featureInteractionFlag = _featureInteractionFlag,
      dataPrepCachingFlag = _dataPrepCachingFlag,
      dataPrepParallelism = _dataPrepParallelism,
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
      featureInteractionConfig = _featureInteractionConfig,
      geneticConfig = _geneticConfig,
      mlFlowLoggingFlag = _mlFlowLoggingFlag,
      mlFlowLogArtifactsFlag = _mlFlowArtifactsFlag,
      mlFlowConfig = _mlFlowConfig,
      inferenceConfigSaveLocation = _inferenceConfigSaveLocation,
      dataReductionFactor = _dataReductionFactor,
      pipelineDebugFlag = _pipelineDebugFlag
    )
    this
  }

  private def setFillConfig(config: FillConfig): this.type = {

    _fillConfig = config
    _numericFillStat = config.numericFillStat
    _characterFillStat = config.characterFillStat
    _modelSelectionDistinctThreshold = config.modelSelectionDistinctThreshold
    _cardinalitySwitchFlag = config.cardinalitySwitch
    _cardinalityType = config.cardinalityType
    _cardinalityLimit = config.cardinalityLimit
    _cardinalityPrecision = config.cardinalityPrecision
    _cardinalityCheckMode = config.cardinalityCheckMode
    _naFillFilterPrecision = config.filterPrecision
    _categoricalNAFillMap = config.categoricalNAFillMap
    _numericNAFillMap = config.numericNAFillMap
    _characterNABlanketFillValue = config.characterNABlanketFillValue
    _numericNABlanketFillValue = config.numericNABlanketFillValue
    _naFillMode = config.naFillMode

    this
  }

  private def setOutlierConfig(config: OutlierConfig): this.type = {

    _outlierConfig = config
    _filterBounds = config.filterBounds
    _lowerFilterNTile = config.lowerFilterNTile
    _upperFilterNTile = config.upperFilterNTile
    _filterPrecision = config.filterPrecision
    _continuousDataThreshold = config.continuousDataThreshold
    _fieldsToIgnore = config.fieldsToIgnore

    this
  }

  private def setPearsonConfig(config: PearsonConfig): this.type = {

    _pearsonConfig = config
    _pearsonFilterStatistic = config.filterStatistic
    _pearsonFilterDirection = config.filterDirection
    _pearsonFilterManualValue = config.filterManualValue
    _pearsonFilterMode = config.filterMode
    _pearsonAutoFilterNTile = config.autoFilterNTile

    this
  }

  private def setCovarianceConfig(config: CovarianceConfig): this.type = {
    _covarianceConfig = config
    _correlationCutoffLow = config.correlationCutoffLow
    _correlationCutoffHigh = config.correlationCutoffHigh

    this
  }

  private def setScalerConfig(config: ScalingConfig): this.type = {

    _scalingConfig = config
    _scalerType = config.scalerType
    _scalerMin = config.scalerMin
    _scalerMax = config.scalerMax
    _standardScalerMeanFlag = config.standardScalerMeanFlag
    _standardScalerStdDevFlag = config.standardScalerStdDevFlag
    _pNorm = config.pNorm

    this
  }

  private def setFeatureInteractionConfig(
    config: FeatureInteractionConfig
  ): this.type = {

    _featureInteractionConfig = config
    _featureInteractionRetentionMode = config.retentionMode
    _featureInteractionContinuousDiscretizerBucketCount =
      config.continuousDiscretizerBucketCount
    _featureInteractionParallelism = config.parallelism
    _featureInteractionTargetInteractionPercentage =
      config.targetInteractionPercentage

    this
  }

  private def setKSampleConfig(config: KSampleConfig): this.type = {

    _kSampleConfig = config
    _syntheticCol = config.syntheticCol
    _kGroups = config.kGroups
    _kMeansMaxIter = config.kMeansMaxIter
    _kMeansTolerance = config.kMeansTolerance
    _kMeansDistanceMeasurement = config.kMeansDistanceMeasurement
    _kMeansSeed = config.kMeansSeed
    _kMeansPredictionCol = config.kMeansPredictionCol
    _lshHashTables = config.lshHashTables
    _lshSeed = config.lshSeed
    _lshOutputCol = config.lshOutputCol
    _quorumCount = config.quorumCount
    _minimumVectorCountToMutate = config.minimumVectorCountToMutate
    _vectorMutationMethod = config.vectorMutationMethod
    _mutationMode = config.mutationMode
    _mutationValue = config.mutationValue
    _labelBalanceMode = config.labelBalanceMode
    _cardinalityThreshold = config.cardinalityThreshold
    _numericRatio = config.numericRatio
    _numericTarget = config.numericTarget
    _outputDfRepartitionScaleFactor = config.outputDfRepartitionScaleFactor

    this
  }

  private def setFirstGenerationConfig(
    config: FirstGenerationConfig
  ): this.type = {

    _firstGenerationConfig = config
    _firstGenerationPermutationCount = config.permutationCount
    _firstGenerationIndexMixingMode = config.indexMixingMode
    _firstGenerationArraySeed = config.arraySeed

    this
  }

  private def setGeneticConfig(config: GeneticConfig): this.type = {

    _geneticConfig = config
    _parallelism = config.parallelism
    _kFold = config.kFold
    _trainPortion = config.trainPortion
    _trainSplitMethod = config.trainSplitMethod
    setKSampleConfig(config.kSampleConfig)
    _trainSplitChronologicalColumn = config.trainSplitChronologicalColumn
    _trainSplitChronologicalRandomPercentage =
      config.trainSplitChronologicalRandomPercentage
    _seed = config.seed
    _firstGenerationGenePool = config.firstGenerationGenePool
    _numberOfGenerations = config.numberOfGenerations
    _numberOfParentsToRetain = config.numberOfParentsToRetain
    _numberOfMutationsPerGeneration = config.numberOfMutationsPerGeneration
    _geneticMixing = config.geneticMixing
    _generationalMutationStrategy = config.generationalMutationStrategy
    _fixedMutationValue = config.fixedMutationValue
    _mutationMagnitudeMode = config.mutationMagnitudeMode
    _evolutionStrategy = config.evolutionStrategy
    _continuousEvolutionMaxIterations = config.continuousEvolutionMaxIterations
    _continuousEvolutionStoppingScore = config.continuousEvolutionStoppingScore
    _continuousEvolutionParallelism = config.continuousEvolutionParallelism
    _continuousEvolutionMutationAggressiveness =
      config.continuousEvolutionMutationAggressiveness
    _continuousEvolutionGeneticMixing = config.continuousEvolutionGeneticMixing
    _continuousEvolutionRollingImprovementCount =
      config.continuousEvolutionRollingImprovementCount
    _modelSeedMap = config.modelSeed
    _hyperSpaceInference = config.hyperSpaceInference
    _hyperSpaceInferenceCount = config.hyperSpaceInferenceCount
    _hyperSpaceModelType = config.hyperSpaceModelType
    _hyperSpaceModelCount = config.hyperSpaceModelCount
    _firstGenerationMode = config.initialGenerationMode
    _continuousEvolutionImprovementThreshold =
      config.continuousEvolutionImprovementThreshold
    _geneticMBORegressorType = config.geneticMBORegressorType
    _geneticMBOCandidateFactor = config.geneticMBOCandidateFactor
    setFirstGenerationConfig(config.initialGenerationConfig)

    this
  }

  private def resetMlFlowConfig(config: MLFlowConfig): this.type = {

    _mlFlowConfig = config
    _mlFlowTrackingURI = config.mlFlowTrackingURI
    _mlFlowExperimentName = config.mlFlowExperimentName
    _mlFlowAPIToken = config.mlFlowAPIToken
    _mlFlowModelSaveDirectory = config.mlFlowModelSaveDirectory
    _mlFlowLoggingMode = config.mlFlowLoggingMode
    _mlFlowBestSuffix = config.mlFlowBestSuffix
    _mlFlowCustomRunTags = config.mlFlowCustomRunTags

    this
  }

  def setMainConfig(value: MainConfig): this.type = {
    _mainConfig = value

    /**
      * Reset all of the local var's so that setters can be used in a chained manner without reverting to defaults.
      */
    _modelingFamily = value.modelFamily
    _labelCol = value.labelCol
    _featuresCol = value.featuresCol
    _naFillFlag = value.naFillFlag
    _varianceFilterFlag = value.varianceFilterFlag
    _outlierFilterFlag = value.outlierFilterFlag
    _pearsonFilterFlag = value.pearsonFilteringFlag
    _covarianceFilterFlag = value.covarianceFilteringFlag
    _oneHotEncodeFlag = value.oneHotEncodeFlag
    _scalingFlag = value.scalingFlag
    _featureInteractionFlag = value.featureInteractionFlag
    _dataPrepCachingFlag = value.dataPrepCachingFlag
    _dataPrepParallelism = value.dataPrepParallelism
    _autoStoppingFlag = value.autoStoppingFlag
    _autoStoppingScore = value.autoStoppingScore
    _featureImportanceCutoffType = value.featureImportanceCutoffType
    _featureImportanceCutoffValue = value.featureImportanceCutoffValue
    _dateTimeConversionType = value.dateTimeConversionType
    _fieldsToIgnoreInVector = value.fieldsToIgnoreInVector
    _numericBoundaries = value.numericBoundaries
    _stringBoundaries = value.stringBoundaries
    _scoringMetric = value.scoringMetric
    _scoringOptimizationStrategy = value.scoringOptimizationStrategy
    setFillConfig(value.fillConfig)
    setOutlierConfig(value.outlierConfig)
    setPearsonConfig(value.pearsonConfig)
    setCovarianceConfig(value.covarianceConfig)
    setScalerConfig(value.scalingConfig)
    setFeatureInteractionConfig(value.featureInteractionConfig)
    setGeneticConfig(value.geneticConfig)
    _mlFlowLoggingFlag = value.mlFlowLoggingFlag
    _mlFlowArtifactsFlag = value.mlFlowLogArtifactsFlag
    resetMlFlowConfig(value.mlFlowConfig)
    _inferenceConfigSaveLocation = value.inferenceConfigSaveLocation
    _dataReductionFactor = value.dataReductionFactor
    _pipelineDebugFlag = value.pipelineDebugFlag

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
      oneHotEncodeFlag = _oneHotEncodeFlag,
      scalingFlag = _scalingFlag,
      featureInteractionFlag = _featureInteractionFlag,
      dataPrepCachingFlag = _dataPrepCachingFlag,
      dataPrepParallelism = _dataPrepParallelism,
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
      featureInteractionConfig = _featureInteractionConfig,
      geneticConfig = _geneticConfig,
      mlFlowLoggingFlag = _mlFlowLoggingFlag,
      mlFlowLogArtifactsFlag = _mlFlowArtifactsFlag,
      mlFlowConfig = _mlFlowConfig,
      inferenceConfigSaveLocation = _inferenceConfigSaveLocation,
      dataReductionFactor = _dataReductionFactor,
      pipelineDebugFlag = _pipelineDebugFlag
    )
    this
  }

  def setFeatConfig(value: MainConfig): this.type = {
    _featureImportancesConfig = value
    require(
      value.modelFamily == "RandomForest",
      s"Model Family for Feature Importances must be 'RandomForest'. ${value.modelFamily} is not supported."
    )
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
      oneHotEncodeFlag = _oneHotEncodeFlag,
      scalingFlag = _scalingFlag,
      featureInteractionFlag = _featureInteractionFlag,
      dataPrepCachingFlag = _dataPrepCachingFlag,
      dataPrepParallelism = _dataPrepParallelism,
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
      featureInteractionConfig = _featureInteractionConfig,
      geneticConfig = _geneticConfig,
      mlFlowLoggingFlag = _mlFlowLoggingFlag,
      mlFlowLogArtifactsFlag = _mlFlowArtifactsFlag,
      mlFlowConfig = _mlFlowConfig,
      inferenceConfigSaveLocation = _inferenceConfigSaveLocation,
      dataReductionFactor = _dataReductionFactor,
      pipelineDebugFlag = _pipelineDebugFlag
    )
    this
  }

  def setTreeSplitsConfig(value: MainConfig): this.type = {
    _treeSplitsConfig = value
    require(
      value.modelFamily == "Trees",
      s"Model Family for Trees Splits must be 'Trees'. ${value.modelFamily} is not supported."
    )
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

  def getOneHotEncodingStatus: Boolean = _oneHotEncodeFlag

  def getScalingStatus: Boolean = _scalingFlag

  def getFeatureInteractionStatus: Boolean = _featureInteractionFlag

  def getDataPrepCachingStatus: Boolean = _dataPrepCachingFlag

  def getDataPrepParallelism: Int = _dataPrepParallelism

  def getNumericBoundaries: Map[String, (Double, Double)] = _numericBoundaries

  def getStringBoundaries: Map[String, List[String]] = _stringBoundaries

  def getScoringMetric: String = _scoringMetric

  def getScoringOptimizationStrategy: String = _scoringOptimizationStrategy

  def getNumericFillStat: String = _numericFillStat

  def getCharacterFillStat: String = _characterFillStat

  def getDateTimeConversionType: String = _dateTimeConversionType

  def getFieldsToIgnoreInVector: Array[String] = _fieldsToIgnoreInVector

  def getNAFillFilterPrecision: Double = _naFillFilterPrecision

  def getCategoricalNAFillMap: Map[String, String] = _categoricalNAFillMap

  def getNumericNAFillMap: Map[String, AnyVal] = _numericNAFillMap

  def getCharacterNABlanketFillValue: String = _characterNABlanketFillValue

  def getNumericNABlanketFillValue: Double = _numericNABlanketFillValue

  def getNAFillMode: String = _naFillMode

  def getCardinalitySwitch: Boolean = _cardinalitySwitchFlag

  def getCardinalityType: String = _cardinalityType

  def getCardinalityLimit: Int = _cardinalityLimit

  def getCardinalityPrecision: Double = _cardinalityPrecision

  def getCardinalityCheckMode: String = _cardinalityCheckMode

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

  def getFeatureInteractionConfig: FeatureInteractionConfig =
    _featureInteractionConfig

  def getFeatureInteractionRetentionMode: String =
    _featureInteractionRetentionMode

  def getFeatureInteractionContinuousDiscretizerBucketCount: Int =
    _featureInteractionContinuousDiscretizerBucketCount

  def getFeatureInteractionParallelism: Int = _featureInteractionParallelism

  def getFeatureInteractionTargetInteractionPercentage: Double =
    _featureInteractionTargetInteractionPercentage

  def getParallelism: Int = _parallelism

  def getKFold: Int = _kFold

  def getTrainPortion: Double = _trainPortion

  def getTrainSplitMethod: String = _trainSplitMethod

  def getKSampleConfig: KSampleConfig = _kSampleConfig

  def getSyntheticCol: String = _syntheticCol

  def getKGroups: Int = _kGroups

  def getKMeansMaxIter: Int = _kMeansMaxIter

  def getKMeansTolerance: Double = _kMeansTolerance

  def getKMeansDistanceMeasurement: String = _kMeansDistanceMeasurement

  def getKMeansSeed: Long = _kMeansSeed

  def getKMeansPredictionCol: String = _kMeansPredictionCol

  def getLSHHashTables: Int = _lshHashTables

  def getLSHOutputCol: String = _lshOutputCol

  def getQuorumCount: Int = _quorumCount

  def getMinimumVectorCountToMutate: Int = _minimumVectorCountToMutate

  def getVectorMutationMethod: String = _vectorMutationMethod

  def getMutationMode: String = _mutationMode

  def getMutationValue: Double = _mutationValue

  def getTrainSplitChronologicalColumn: String = _trainSplitChronologicalColumn

  def getTrainSplitChronologicalRandomPercentage: Double =
    _trainSplitChronologicalRandomPercentage

  def getSeed: Long = _seed

  def getFirstGenerationGenePool: Int = _firstGenerationGenePool

  def getNumberOfGenerations: Int = _numberOfGenerations

  def getNumberOfParentsToRetain: Int = _numberOfParentsToRetain

  def getNumberOfMutationsPerGeneration: Int = _numberOfMutationsPerGeneration

  def getGeneticMixing: Double = _geneticMixing

  def getGenerationalMutationStrategy: String = _generationalMutationStrategy

  def getFixedMutationValue: Int = _fixedMutationValue

  def getMutationMagnitudeMode: String = _mutationMagnitudeMode

  def getModelSeedSetStatus: Boolean = _modelSeedSetStatus

  def getModelSeedMap: Map[String, Any] = _modelSeedMap

  def getFirstGenerationPermutationCount: Int = _firstGenerationPermutationCount

  def getFirstGenerationIndexMixingMode: String =
    _firstGenerationIndexMixingMode

  def getFirstGenerationArraySeed: Long = _firstGenerationArraySeed

  def getHyperSpaceInferenceStatus: Boolean = _hyperSpaceInference

  def getHyperSpaceInferenceCount: Int = _hyperSpaceInferenceCount

  def getHyperSpaceModelType: String = _hyperSpaceModelType

  def getHyperSpaceModelCount: Int = _hyperSpaceModelCount

  def getFirstGenerationConfig: FirstGenerationConfig = _firstGenerationConfig

  def getFirstGenerationMode: String = _firstGenerationMode

  def getMlFlowLoggingFlag: Boolean = _mlFlowLoggingFlag

  def getMlFlowLogArtifactsFlag: Boolean = _mlFlowArtifactsFlag

  def getMlFlowTrackingURI: String = _mlFlowTrackingURI

  def getMlFlowExperimentName: String = _mlFlowExperimentName

  def getMlFlowModelSaveDirectory: String = _mlFlowModelSaveDirectory

  def getMlFlowLoggingMode: String = _mlFlowLoggingMode

  def getMlFlowBestSuffix: String = _mlFlowBestSuffix

  def getMlFlowCustomRunTags: Map[String, String] = _mlFlowCustomRunTags

  def getMlFlowConfig: MLFlowConfig = _mlFlowConfig

  def getGeneticConfig: GeneticConfig = _geneticConfig

  def getMainConfig: MainConfig = _mainConfig

  def getFeatConfig: MainConfig = _featureImportancesConfig

  def getTreeSplitsConfig: MainConfig = _treeSplitsConfig

  def getAutoStoppingFlag: Boolean = _autoStoppingFlag

  def getAutoStoppingScore: Double = _autoStoppingScore

  def getFeatureImportanceCutoffType: String = _featureImportanceCutoffType

  def getFeatureImportanceCutoffValue: Double = _featureImportanceCutoffValue

  def getEvolutionStrategy: String = _evolutionStrategy

  def getContinuousEvolutionMaxIterations: Int =
    _continuousEvolutionMaxIterations

  def getContinuousEvolutionStoppingScore: Double =
    _continuousEvolutionStoppingScore

  def getContinuousEvolutionParallelism: Int = _continuousEvolutionParallelism

  def getContinuousEvolutionMutationAggressiveness: Int =
    _continuousEvolutionMutationAggressiveness

  def getContinuousEvolutionGeneticMixing: Double =
    _continuousEvolutionGeneticMixing

  def getContinuousEvolutionRollingImporvementCount: Int =
    _continuousEvolutionRollingImprovementCount

  def getInferenceConfigSaveLocation: String = _inferenceConfigSaveLocation

  def getDataReductionFactor: Double = _dataReductionFactor

  /**
    * Helper method for extracting the config from a run's GenericModelReturn payload
    * This is designed to handle "lazy" copy/paste from either stdout or the mlflow ui.
    * The alternative (preferred method of seeding a run start) is to submit a Map() for the run configuration seed.
    *
    * @param fullModelReturn: String The Generic Model Config of a run, to be used as a starting point for further
    *                       tuning or refinement.
    * @return A Map Object that can be parsed into the requisite case class definition to set a seed for a particular
    *         type of model run.
    */
  private def extractGenericModelReturnMap(
    fullModelReturn: String
  ): Map[String, Any] = {

    val patternToMatch = "(?<=\\()[^()]*".r

    val configElements =
      patternToMatch.findAllIn(fullModelReturn).toList(1).split(",")

    var configMap = Map[String, Any]()

    configElements.foreach { x =>
      val components = x.trim.split(" -> ")
      configMap += (components(0) -> components(1))
    }
    configMap
  }

}
