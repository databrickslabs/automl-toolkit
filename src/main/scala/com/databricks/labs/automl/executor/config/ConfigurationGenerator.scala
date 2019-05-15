package com.databricks.labs.automl.executor.config

import com.databricks.labs.automl.params._
import org.json4s.{Formats, NoTypeHints}
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.{writePretty, read}


class GenericConfigGenerator(predictionType: String) extends ConfigurationDefaults {

  import PredictionType._

  private val familyType: PredictionType = predictionTypeEvaluator(predictionType)

  private var _genericConfig = genericConfig(familyType)


  def setLabelCol(value: String): this.type = {
    _genericConfig.labelCol = value
    this
  }

  def setFeaturesCol(value: String): this.type = {
    _genericConfig.featuresCol = value
    this
  }

  def setDateTimeConversionType(value: String): this.type = {
    validateMembership(value, allowableDateTimeConversionTypes, "DateTimeConversionType")
    _genericConfig.dateTimeConversionType = value
    this
  }

  def setFieldsToIgnoreInVector(value: Array[String]): this.type = {
    _genericConfig.fieldsToIgnoreInVector = value
    this
  }

  def setScoringMetric(value: String): this.type = {
    familyType match {
      case Regressor =>
        validateMembership(value, allowableRegressionScoringMetrics, s"$predictionType Scoring Metric")
      case Classifier =>
        validateMembership(value, allowableClassificationScoringMetrics, s"$predictionType Scoring Metric")
    }
    _genericConfig.scoringMetric = value
    this
  }

  def setScoringOptimizationStrategy(value: String): this.type = {
    validateMembership(value, allowableScoringOptimizationStrategies, "ScoringOptimizationStrategy")
    _genericConfig.scoringOptimizationStrategy = value
    this
  }

  def getLabelCol: String = _genericConfig.labelCol

  def getFeaturesCol: String = _genericConfig.featuresCol

  def getDateTimeConversionType: String = _genericConfig.dateTimeConversionType

  def getFieldsToIgnoreInVector: Array[String] = _genericConfig.fieldsToIgnoreInVector

  def getScoringMetric: String = _genericConfig.scoringMetric

  def getScoringOptimizationStrategy: String = _genericConfig.scoringOptimizationStrategy

  def getConfig: GenericConfig = _genericConfig

}

object GenericConfigGenerator {

  def generateDefaultClassifierConfig: GenericConfig = new GenericConfigGenerator("classifier").getConfig

  def generateDefaultRegressorConfig: GenericConfig = new GenericConfigGenerator("regressor").getConfig
}

class ConfigurationGenerator(modelFamily: String, predictionType: String, var genericConfig: GenericConfig)
  extends ConfigurationDefaults {

  import FamilyValidator._
  import ModelSelector._

  private val modelType: ModelSelector = modelTypeEvaluator(modelFamily, predictionType)
  private val family: FamilyValidator = familyTypeEvaluator(modelFamily)

  /**
    * Default configuration generation
    */

  private var _instanceConfig = instanceConfig(modelFamily, predictionType)

  /**
    * Switch Config
    */

  def naFillOn(): this.type = {
    _instanceConfig.switchConfig.naFillFlag = true
    this
  }

  def naFillOff(): this.type = {
    _instanceConfig.switchConfig.naFillFlag = false
    this
  }

  def varianceFilterOn(): this.type = {
    _instanceConfig.switchConfig.varianceFilterFlag = true
    this
  }

  def varianceFilterOff(): this.type = {
    _instanceConfig.switchConfig.varianceFilterFlag = false
    this
  }

  def outlierFilterOn(): this.type = {
    _instanceConfig.switchConfig.outlierFilterFlag = true
    this
  }

  def outlierFilterOff(): this.type = {
    _instanceConfig.switchConfig.outlierFilterFlag = false
    this
  }

  def pearsonFilterOn(): this.type = {
    _instanceConfig.switchConfig.pearsonFilterFlag = true
    this
  }

  def pearsonFilterOff(): this.type = {
    _instanceConfig.switchConfig.pearsonFilterFlag = false
    this
  }

  def covarianceFilterOn(): this.type = {
    _instanceConfig.switchConfig.covarianceFilterFlag = true
    this
  }

  def covarianceFilterOff(): this.type = {
    _instanceConfig.switchConfig.covarianceFilterFlag = false
    this
  }

  def oneHotEncodeOn(): this.type = {
    family match {
      case Trees => println("WARNING! OneHotEncoding set on a trees algorithm will likely create a poor model.  " +
        "Proceed at your own risk!")
    }
    _instanceConfig.switchConfig.oneHotEncodeFlag = true
    this
  }

  def oneHotEncodeOff(): this.type = {
    _instanceConfig.switchConfig.oneHotEncodeFlag = false
    this
  }

  def scalingOn(): this.type = {
    _instanceConfig.switchConfig.scalingFlag = true
    this
  }

  def scalingOff(): this.type = {
    _instanceConfig.switchConfig.scalingFlag = false
    this
  }

  def dataPrepCachingOn(): this.type = {
    _instanceConfig.switchConfig.dataPrepCachingFlag = true
    this
  }

  def dataPrepCachingOff(): this.type = {
    _instanceConfig.switchConfig.dataPrepCachingFlag = false
    this
  }

  def autoStoppingOn(): this.type = {
    _instanceConfig.switchConfig.autoStoppingFlag = true
    this
  }

  def autoStoppingOff(): this.type = {
    _instanceConfig.switchConfig.autoStoppingFlag = false
    this
  }


  /**
    * Feature Engineering Config
    */

  def setFillConfigNumericFillStat(value: String): this.type = {
    validateMembership(value, allowableNumericFillStats, "FillConfigNumericFillStat")
    _instanceConfig.featureEngineeringConfig.numericFillStat = value
    this
  }

  def setFillConfigCharacterFillStat(value: String): this.type = {
    validateMembership(value, allowableCharacterFillStats, "FillConfigCharacterFillStat")
    _instanceConfig.featureEngineeringConfig.characterFillStat = value
    this
  }

  //TODO: In the new world of this config, this POJO and its underlying methodology needs to be expunged.

  def setFillConfigModelSelectionDistinctThreshold(value: Int): this.type = {
    _instanceConfig.featureEngineeringConfig.modelSelectionDistinctThreshold = value
    this
  }

  def setOutlierFilterBounds(value: String): this.type = {
    validateMembership(value, allowableOutlierFilterBounds, "OutlierFilterBounds")
    _instanceConfig.featureEngineeringConfig.outlierFilterBounds = value
    this
  }

  def setOutlierLowerFilterNTile(value: Double): this.type = {
    zeroToOneValidation(value, "OutlierLowerFilterNTile")
    _instanceConfig.featureEngineeringConfig.outlierLowerFilterNTile = value
    this
  }

  def setOutlierUpperFilterNTile(value: Double): this.type = {
    zeroToOneValidation(value, "OutlierUpperFilterNTile")
    _instanceConfig.featureEngineeringConfig.outlierUpperFilterNTile = value
    this
  }

  def setOutlierFilterPrecision(value: Double): this.type = {
    if (value == 0.0) println("Warning! Precision of 0 is an exact calculation of quantiles and may not be performant!")
    _instanceConfig.featureEngineeringConfig.outlierFilterPrecision = value
    this
  }

  def setOutlierContinuousDataThreshold(value: Int): this.type = {
    if (value < 50) println("Warning! Values less than 50 may indicate oridinal data!")
    _instanceConfig.featureEngineeringConfig.outlierContinuousDataThreshold = value
    this
  }

  def setOutlierFieldsToIgnore(value: Array[String]): this.type = {
    _instanceConfig.featureEngineeringConfig.outlierFieldsToIgnore = value
    this
  }

  def setPearsonFilterStatistic(value: String): this.type = {
    validateMembership(value, allowablePearsonFilterStats, "PearsonFilterStatistic")
    _instanceConfig.featureEngineeringConfig.pearsonFilterStatistic = value
    this
  }

  def setPearsonFilterDirection(value: String): this.type = {
    validateMembership(value, allowablePearsonFilterDirections, "PearsonFilterDirection")
    _instanceConfig.featureEngineeringConfig.pearsonFilterDirection = value
    this
  }

  def setPearsonFilterManualValue(value: Double): this.type = {
    _instanceConfig.featureEngineeringConfig.pearsonFilterManualValue = value
    this
  }

  def setPearsonFilterMode(value: String): this.type = {
    validateMembership(value, allowablePearsonFilterModes, "PearsonFilterMode")
    _instanceConfig.featureEngineeringConfig.pearsonFilterMode = value
    this
  }

  def setPearsonAutoFilterNTile(value: Double): this.type = {
    zeroToOneValidation(value, "PearsonAutoFilterNTile")
    _instanceConfig.featureEngineeringConfig.pearsonAutoFilterNTile = value
    this
  }

  def setCovarianceCutoffLow(value: Double): this.type = {
    require(value > -1.0, s"Covariance Cutoff Low value $value is outside of allowable range.  Value must be " +
      s"greater than -1.0.")
    _instanceConfig.featureEngineeringConfig.covarianceCorrelationCutoffLow = value
    this
  }

  def setCovarianceCutoffHigh(value: Double): this.type = {
    require(value < 1.0, s"Covariance Cutoff High value $value is outside of allowable range.  Value must be " +
      s"less than 1.0.")
    _instanceConfig.featureEngineeringConfig.covarianceCorrelationCutoffHigh = value
    this
  }

  def setScalingType(value: String): this.type = {
    validateMembership(value, allowableScalers, "ScalingType")
    _instanceConfig.featureEngineeringConfig.scalingType = value
    this
  }

  def setScalingMin(value: Double): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingMin = value
    this
  }

  def setScalingMax(value: Double): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingMax = value
    this
  }

  def setScalingStandardMeanFlagOn(): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingStandardMeanFlag = true
    this
  }

  def setScalingStandardMeanFlagOff(): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingStandardMeanFlag = false
    this
  }

  def setScalingStdDevFlagOn(): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingStdDevFlag = true
    this
  }

  def setScalingStdDevFlagOff(): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingStdDevFlag = false
    this
  }

  def setScalingPNorm(value: Double): this.type = {
    require(value >= 1.0, s"pNorm value: $value is invalid. Value must be greater than or equal to 1.0.")
    _instanceConfig.featureEngineeringConfig.scalingPNorm = value
    this
  }

  def setFeatureImportanceCutoffType(value: String): this.type = {
    validateMembership(value, allowableFeatureImportanceCutoffTypes, "FeatureImportanceCufoffType")
    _instanceConfig.featureEngineeringConfig.featureImportanceCutoffType = value
    this
  }

  def setFeatureImportanceCutoffValue(value: Double): this.type = {
    _instanceConfig.featureEngineeringConfig.featureImportanceCutoffValue = value
    this
  }

  def setDataReductionFactor(value: Double): this.type = {
    zeroToOneValidation(value, "DateReductionFactor")
    _instanceConfig.featureEngineeringConfig.dataReductionFactor = value
    this
  }

  /**
    * Algorithm Config
    */

  def setStringBoundaries(value: Map[String, List[String]]): this.type = {
    validateStringBoundariesKeys(modelType, value)
    _instanceConfig.algorithmConfig.stringBoundaries = value
    this
  }

  def setNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    validateNumericBoundariesValues(value)
    validateNumericBoundariesKeys(modelType, value)
    _instanceConfig.algorithmConfig.numericBoundaries = value
    this
  }

  /**
    * Tuner Config
    */

  def setTunerAutoStoppingScore(value: Double): this.type = {
    _instanceConfig.tunerConfig.tunerAutoStoppingScore = value
    this
  }

  def setTunerParallelism(value: Int): this.type = {
    if (value > 30) println("WARNING - Setting Tuner Parallelism greater than 30 could put excessive stress on the " +
      "Driver.  Ensure driver is monitored for stability.")
    _instanceConfig.tunerConfig.tunerParallelism = value
    this
  }

  def setTunerKFold(value: Int): this.type = {
    if (value < 5) println("WARNING - Setting KFold < 5 may result in a poorly generalized tuning run due to " +
      "over-fitting within a particular train/test split.")
    _instanceConfig.tunerConfig.tunerKFold = value
    this
  }

  def setTunerTrainPortion(value: Double): this.type = {
    require(value > 0.0 & value < 1.0, s"TunerTrainPortion must be within the range of 0.0 to 1.0.")
    if (value < 0.5) println(s"WARNING - setting TunerTrainPortion below 0.5 may result in a poorly fit model.  Best" +
      s" practices guidance typically adheres to a 0.7 or 0.8 test/train ratio.")
    _instanceConfig.tunerConfig.tunerTrainPortion = value
    this
  }

  def setTunerTrainSplitMethod(value: String): this.type = {
    validateMembership(value, allowableTrainSplitMethods, "TunerTrainSplitMethod")
    _instanceConfig.tunerConfig.tunerTrainSplitMethod = value
    this
  }

  def setTunerTrainSplitChronologicalColumn(value: String): this.type = {
    _instanceConfig.tunerConfig.tunerTrainSplitChronologicalColumn = value
    val updatedFieldsToIgnore = genericConfig.fieldsToIgnoreInVector ++: Array(value)
    genericConfig.fieldsToIgnoreInVector = updatedFieldsToIgnore
    this
  }

  def setTunerTrainSplitChronologicalRandomPercentage(value: Double): this.type = {
    if (value > 10) println("[WARNING] TunerTrainSplitChronologicalRandomPercentage setting this value above 10 " +
      "percent will cause significant per-run train/test skew and variability in row counts during training.  " +
      "Use higher values only if this is desired.")
    _instanceConfig.tunerConfig.tunerTrainSplitChronologicalRandomPercentage = value
    this
  }

  def setTunerSeed(value: Long): this.type = {
    _instanceConfig.tunerConfig.tunerSeed = value
    this
  }

  def setTunerFirstGenerationGenePool(value: Int): this.type = {
    if (value < 10) println("[WARNING] TunerFirstGenerationGenePool values of less than 10 may not find global minima" +
      "for hyperparameters.  Consider setting the value > 30 for best performance.")
    _instanceConfig.tunerConfig.tunerFirstGenerationGenePool = value
    this
  }

  def setTunerNumberOfGenerations(value: Int): this.type = {
    if (value < 3) println("[WARNING] TunerNumberOfGenerations set below 3 may not explore hyperparameter feature " +
      "space effectively to arrive at a global minima.")
    if (value > 20) println("[WARNING] TunerNumberOfGenerations set above 20 will take a long time to run.  Evaluate" +
      "whether first generation gene pool count and numer of mutations per generation should be adjusted higher" +
      "instead.")
    _instanceConfig.tunerConfig.tunerNumberOfGenerations = value
    this
  }

  def setTunerNumberOfParentsToRetain(value: Int): this.type = {
    require(value > 0, s"TunerNumberOfParentsToRetain must be > 0. $value is outside of bounds.")
    _instanceConfig.tunerConfig.tunerNumberOfParentsToRetain = value
    this
  }

  def setTunerNumberOfMutationPerGeneration(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerNumberOfMutationsPerGeneration = value
    this
  }

  def setTunerGeneticMixing(value: Double): this.type = {
    zeroToOneValidation(value, "TunerGeneticMixing")
    if (value > 0.9) println(s"[WARNING] Setting TunerGeneticMixing to a value greater than 0.9 will not effectively" +
      s"explore the hyperparameter feature space.  Use such settings only for fine-tuning around a pre-calculated " +
      s"global minima.")
    _instanceConfig.tunerConfig.tunerGeneticMixing = value
    this
  }

  def setTunerGenerationalMutationStrategy(value: String): this.type = {
    validateMembership(value, allowableMutationStrategies, "TunerGenerationalMutationStrategy")
    _instanceConfig.tunerConfig.tunerGenerationalMutationStrategy = value
    this
  }

  def setTunerFixedMutationValue(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerFixedMutationValue = value
    this
  }

  def setTunerMutationMagnitudeMode(value: String): this.type = {
    validateMembership(value, allowableMutationMagnitudeMode, "TunerMutationMagnitudeMode")
    _instanceConfig.tunerConfig.tunerMutationMagnitudeMode = value
    this
  }

  def setTunerEvolutionStrategy(value: String): this.type = {
    validateMembership(value, allowableEvolutionStrategies, "TunerEvolutionStrategy")
    _instanceConfig.tunerConfig.tunerEvolutionStrategy = value
    this
  }

  def setTunerContinuousEvolutionMaxIterations(value: Int): this.type = {
    if (value > 500) println(s"[WARNING] Setting this value higher increases runtime by O(n/parallelism) amount.  " +
      s"Values higher than 500 may take an unacceptably long time to run. ")
    _instanceConfig.tunerConfig.tunerContinuousEvolutionMaxIterations = value
    this
  }

  def setTunerContinuousEvolutionStoppingScore(value: Double): this.type = {
    zeroToOneValidation(value, "TunerContinuuousEvolutionStoppingScore")
    _instanceConfig.tunerConfig.tunerContinuousEvolutionStoppingScore = value
    this
  }

  def setTunerContinuousEvolutionParallelism(value: Int): this.type = {
    if (value > 10) println("[WARNING] Setting value of TunerContinuousEvolutionParallelism greater than 10 may have" +
      "unintended side-effects of a longer convergence time due to async Futures that have not returned results" +
      "by the time that the next iteration is initiated.  Recommended settings are in the range of [4:8] for " +
      "continuous mode.")
    _instanceConfig.tunerConfig.tunerContinuousEvolutionParallelism = value
    this
  }

  def setTunerContinuousEvolutionMutationAggressiveness(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerContinuousEvolutionMutationAggressiveness = value
    this
  }

  def setTunerContinuousEvolutionGeneticMixing(value: Double): this.type = {
    zeroToOneValidation(value, "TunerContinuousEvolutionGeneticMixing")
    if (value > 0.9) println(s"[WARNING] Setting TunerContinuousEvolutionGeneticMixing to a value greater than 0.9 " +
      s"will not effectively explore the hyperparameter feature space.  Use such settings only for fine-tuning " +
      s"around a pre-calculated global minima.")
    _instanceConfig.tunerConfig.tunerContinuousEvolutionGeneticMixing = value
    this
  }

  def setTunerContinuousEvolutionRollingImprovementCount(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerContinuousEvolutionRollingImprovingCount = value
    this
  }

  //TODO: per model validation of keys?
  def setTunerModelSeed(value: Map[String, Any]): this.type = {
    _instanceConfig.tunerConfig.tunerModelSeed = value
    this
  }

  def setTunerHyperSpaceInferenceOn(): this.type = {
    _instanceConfig.tunerConfig.tunerHyperSpaceInference = true
    this
  }

  def setTunerHyperSpaceInferenceOff(): this.type = {
    _instanceConfig.tunerConfig.tunerHyperSpaceInference = false
    this
  }

  def setTunerHyperSpaceInferenceCount(value: Int): this.type = {
    if (value > 500000) println("[WARNING] Setting TunerHyperSpaceInferenceCount above 500,000 will put stress on the " +
      "driver for generating so many leaves.")
    if (value > 1000000) throw new UnsupportedOperationException(s"Setting TunerHyperSpaceInferenceCount above " +
      s"1,000,000 is not supported due to runtime considerations.  $value is too large of a value.")
    _instanceConfig.tunerConfig.tunerHyperSpaceInferenceCount = value
    this
  }

  def setTunerHyperSpaceModelCount(value: Int): this.type = {
    if (value > 50) println("[WARNING] TunerHyperSpaceModelCount values set excessively high will incur long runtime" +
      "costs after the conclusion of Genetic Tuner running.  Gains are diminishing after a value of 20.")
    _instanceConfig.tunerConfig.tunerHyperSpaceModelCount = value
    this
  }

  def setTunerHyperSpaceModelType(value: String): this.type = {
    validateMembership(value, allowableHyperSpaceModelTypes, "TunerHyperSpaceModelType")
    _instanceConfig.tunerConfig.tunerHyperSpaceModelType = value
    this
  }

  def setTunerInitialGenerationMode(value: String): this.type = {
    validateMembership(value, allowableInitialGenerationModes, "TunerInitialGenerationMode")
    _instanceConfig.tunerConfig.tunerInitialGenerationMode = value
    this
  }

  def setTunerInitialGenerationPermutationCount(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerInitialGenerationPermutationCount = value
    this
  }

  def setTunerInitialGenerationIndexMixingMode(value: String): this.type = {
    validateMembership(value, allowableInitialGenerationIndexMixingModes,
      "TunerInitialGenerationIndexMixingMode")
    _instanceConfig.tunerConfig.tunerInitialGenerationIndexMixingMode = value
    this
  }

  def setTunerInitialGenerationArraySeed(value: Long): this.type = {
    _instanceConfig.tunerConfig.tunerInitialGenerationArraySeed = value
    this
  }


  /**
    * MLFlow Logging Config
    */

  def setMlFlowLoggingOn(): this.type = {
    _instanceConfig.loggingConfig.mlFlowLoggingFlag = true
    this
  }

  def setMlFlowLoggingOff(): this.type = {
    _instanceConfig.loggingConfig.mlFlowLoggingFlag = false
    this
  }

  def setMlFlowLoggingFlag(value: Boolean): this.type = {
    _instanceConfig.loggingConfig.mlFlowLoggingFlag = value
    this
  }

  def setMlFlowLogArtifactsOn(): this.type = {
    _instanceConfig.loggingConfig.mlFlowLogArtifactsFlag = true
    this
  }

  def setMlFlowLogArtifactsOff(): this.type = {
    _instanceConfig.loggingConfig.mlFlowLogArtifactsFlag = false
    this
  }

  def setMlFlowLogArtifactsFlag(value: Boolean): this.type = {
    _instanceConfig.loggingConfig.mlFlowLogArtifactsFlag = value
    this
  }

  //TODO: Add path validation here!!
  def setMlFlowTrackingURI(value: String): this.type = {
    _instanceConfig.loggingConfig.mlFlowTrackingURI = value
    this
  }

  def setMlFlowExperimentName(value: String): this.type = {
    _instanceConfig.loggingConfig.mlFlowExperimentName = value
    this
  }

  def setMlFlowAPIToken(value: String): this.type = {
    _instanceConfig.loggingConfig.mlFlowAPIToken = value
    this
  }

  def setMlFlowModelSaveDirectory(value: String): this.type = {
    _instanceConfig.loggingConfig.mlFlowModelSaveDirectory = value
    this
  }

  def setMlFlowLoggingMode(value: String): this.type = {
    validateMembership(value, allowableMlFlowLoggingModes, "MlFlowLoggingMode")
    _instanceConfig.loggingConfig.mlFlowLoggingMode = value
    this
  }

  def setMlFlowBestSuffix(value: String): this.type = {
    _instanceConfig.loggingConfig.mlFlowBestSuffix = value
    this
  }

  def setInferenceConfigSaveLocation(value: String): this.type = {
    _instanceConfig.loggingConfig.inferenceConfigSaveLocation = value
    this
  }

  /**
    * Getters
    */


  def getInstanceConfig: InstanceConfig = _instanceConfig

  def generateMainConfig: MainConfig = ConfigurationGenerator.generateMainConfig(_instanceConfig)

  def generateFeatureImportanceConfig: MainConfig = ConfigurationGenerator.generateMainConfig(_instanceConfig)

  def generateTreeSplitConfig: MainConfig = ConfigurationGenerator.generateMainConfig(_instanceConfig)

}


object ConfigurationGenerator extends ConfigurationDefaults {

  import PredictionType._

  def apply(modelFamily: String, predictionType: String, genericConfig: GenericConfig): ConfigurationGenerator =
    new ConfigurationGenerator(modelFamily, predictionType, genericConfig)

  /**
    *
    * @param modelFamily
    * @param predictionType
    * @return
    */
  def generateDefaultConfig(modelFamily: String, predictionType: String): InstanceConfig = {

    predictionTypeEvaluator(predictionType) match {
      case Regressor => new ConfigurationGenerator(modelFamily, predictionType,
        GenericConfigGenerator.generateDefaultRegressorConfig).getInstanceConfig
      case Classifier => new ConfigurationGenerator(modelFamily, predictionType,
        GenericConfigGenerator.generateDefaultClassifierConfig).getInstanceConfig
    }

  }

  private def standardizeModelFamilyStrings(value: String): String = {
    value match {
      case "randomforest" => "RandomForest"
      case "gbt" => "GBT"
      case "linearregression" => "LinearRegression"
      case "logisticregression" => "LogisticRegression"
      case "mlpc" => "MLPC"
      case "svm" => "SVM"
      case "trees" => "Trees"
      case "xgboost" => "XGBoost"
      case _ => throw new IllegalArgumentException(s"standardizeModelFamilyStrings does not have a supported" +
        s"type of: $value")
    }
  }

  /**
    *
    * @param config
    * @return
    */
  def generateMainConfig(config: InstanceConfig): MainConfig = {
    MainConfig(
      modelFamily = standardizeModelFamilyStrings(config.modelFamily),
      labelCol = config.genericConfig.labelCol,
      featuresCol = config.genericConfig.featuresCol,
      naFillFlag = config.switchConfig.naFillFlag,
      varianceFilterFlag = config.switchConfig.varianceFilterFlag,
      outlierFilterFlag = config.switchConfig.outlierFilterFlag,
      pearsonFilteringFlag = config.switchConfig.pearsonFilterFlag,
      covarianceFilteringFlag = config.switchConfig.covarianceFilterFlag,
      oneHotEncodeFlag = config.switchConfig.oneHotEncodeFlag,
      scalingFlag = config.switchConfig.scalingFlag,
      dataPrepCachingFlag = config.switchConfig.dataPrepCachingFlag,
      autoStoppingFlag = config.switchConfig.autoStoppingFlag,
      autoStoppingScore = config.tunerConfig.tunerAutoStoppingScore,
      featureImportanceCutoffType = config.featureEngineeringConfig.featureImportanceCutoffType,
      featureImportanceCutoffValue = config.featureEngineeringConfig.featureImportanceCutoffValue,
      dateTimeConversionType = config.genericConfig.dateTimeConversionType,
      fieldsToIgnoreInVector = config.genericConfig.fieldsToIgnoreInVector,
      numericBoundaries = config.algorithmConfig.numericBoundaries,
      stringBoundaries = config.algorithmConfig.stringBoundaries,
      scoringMetric = config.genericConfig.scoringMetric,
      scoringOptimizationStrategy = config.genericConfig.scoringOptimizationStrategy,
      fillConfig = FillConfig(
        numericFillStat = config.featureEngineeringConfig.numericFillStat,
        characterFillStat = config.featureEngineeringConfig.characterFillStat,
        modelSelectionDistinctThreshold = config.featureEngineeringConfig.modelSelectionDistinctThreshold
      ),
      outlierConfig = OutlierConfig(
        filterBounds = config.featureEngineeringConfig.outlierFilterBounds,
        lowerFilterNTile = config.featureEngineeringConfig.outlierLowerFilterNTile,
        upperFilterNTile = config.featureEngineeringConfig.outlierUpperFilterNTile,
        filterPrecision = config.featureEngineeringConfig.outlierFilterPrecision,
        continuousDataThreshold = config.featureEngineeringConfig.outlierContinuousDataThreshold,
        fieldsToIgnore = config.featureEngineeringConfig.outlierFieldsToIgnore
      ),
      pearsonConfig = PearsonConfig(
        filterStatistic = config.featureEngineeringConfig.pearsonFilterStatistic,
        filterDirection = config.featureEngineeringConfig.pearsonFilterDirection,
        filterManualValue = config.featureEngineeringConfig.pearsonFilterManualValue,
        filterMode = config.featureEngineeringConfig.pearsonFilterMode,
        autoFilterNTile = config.featureEngineeringConfig.pearsonAutoFilterNTile
      ),
      covarianceConfig = CovarianceConfig(
        correlationCutoffLow = config.featureEngineeringConfig.covarianceCorrelationCutoffLow,
        correlationCutoffHigh = config.featureEngineeringConfig.covarianceCorrelationCutoffHigh
      ),
      scalingConfig = ScalingConfig(
        scalerType = config.featureEngineeringConfig.scalingType,
        scalerMin = config.featureEngineeringConfig.scalingMin,
        scalerMax = config.featureEngineeringConfig.scalingMax,
        standardScalerMeanFlag = config.featureEngineeringConfig.scalingStandardMeanFlag,
        standardScalerStdDevFlag = config.featureEngineeringConfig.scalingStdDevFlag,
        pNorm = config.featureEngineeringConfig.scalingPNorm
      ),
      geneticConfig = GeneticConfig(
        parallelism = config.tunerConfig.tunerParallelism,
        kFold = config.tunerConfig.tunerKFold,
        trainPortion = config.tunerConfig.tunerTrainPortion,
        trainSplitMethod = config.tunerConfig.tunerTrainSplitMethod,
        trainSplitChronologicalColumn = config.tunerConfig.tunerTrainSplitChronologicalColumn,
        trainSplitChronologicalRandomPercentage = config.tunerConfig.tunerTrainSplitChronologicalRandomPercentage,
        seed = config.tunerConfig.tunerSeed,
        firstGenerationGenePool = config.tunerConfig.tunerFirstGenerationGenePool,
        numberOfGenerations = config.tunerConfig.tunerNumberOfGenerations,
        numberOfParentsToRetain = config.tunerConfig.tunerNumberOfParentsToRetain,
        numberOfMutationsPerGeneration = config.tunerConfig.tunerNumberOfMutationsPerGeneration,
        geneticMixing = config.tunerConfig.tunerGeneticMixing,
        generationalMutationStrategy = config.tunerConfig.tunerGenerationalMutationStrategy,
        fixedMutationValue = config.tunerConfig.tunerFixedMutationValue,
        mutationMagnitudeMode = config.tunerConfig.tunerMutationMagnitudeMode,
        evolutionStrategy = config.tunerConfig.tunerEvolutionStrategy,
        continuousEvolutionMaxIterations = config.tunerConfig.tunerContinuousEvolutionMaxIterations,
        continuousEvolutionStoppingScore = config.tunerConfig.tunerContinuousEvolutionStoppingScore,
        continuousEvolutionParallelism = config.tunerConfig.tunerContinuousEvolutionParallelism,
        continuousEvolutionMutationAggressiveness = config.tunerConfig.tunerContinuousEvolutionMutationAggressiveness,
        continuousEvolutionGeneticMixing = config.tunerConfig.tunerContinuousEvolutionGeneticMixing,
        continuousEvolutionRollingImprovementCount = config.tunerConfig.tunerContinuousEvolutionRollingImprovingCount,
        modelSeed = config.tunerConfig.tunerModelSeed,
        hyperSpaceInference = config.tunerConfig.tunerHyperSpaceInference,
        hyperSpaceInferenceCount = config.tunerConfig.tunerHyperSpaceInferenceCount,
        hyperSpaceModelType = config.tunerConfig.tunerHyperSpaceModelType,
        hyperSpaceModelCount = config.tunerConfig.tunerHyperSpaceModelCount,
        initialGenerationMode = config.tunerConfig.tunerInitialGenerationMode,
        initialGenerationConfig = FirstGenerationConfig(
          permutationCount = config.tunerConfig.tunerInitialGenerationPermutationCount,
          indexMixingMode = config.tunerConfig.tunerInitialGenerationIndexMixingMode,
          arraySeed = config.tunerConfig.tunerInitialGenerationArraySeed
        )
      ),
      mlFlowLoggingFlag = config.loggingConfig.mlFlowLoggingFlag,
      mlFlowLogArtifactsFlag = config.loggingConfig.mlFlowLogArtifactsFlag,
      mlFlowConfig = MLFlowConfig(
        mlFlowTrackingURI = config.loggingConfig.mlFlowTrackingURI,
        mlFlowExperimentName = config.loggingConfig.mlFlowExperimentName,
        mlFlowAPIToken = config.loggingConfig.mlFlowAPIToken,
        mlFlowModelSaveDirectory = config.loggingConfig.mlFlowModelSaveDirectory,
        mlFlowLoggingMode = config.loggingConfig.mlFlowLoggingMode,
        mlFlowBestSuffix = config.loggingConfig.mlFlowBestSuffix
      ),
      inferenceConfigSaveLocation = config.loggingConfig.inferenceConfigSaveLocation,
      dataReductionFactor = config.featureEngineeringConfig.dataReductionFactor
    )


  }

  /**
    *
    * @param modelFamily
    * @param predictionType
    * @return
    */
  def generateDefaultMainConfig(modelFamily: String, predictionType: String): MainConfig = {
    val defaultInstanceConfig = generateDefaultConfig(modelFamily, predictionType)
    generateMainConfig(defaultInstanceConfig)
  }

  /**
    *
    * @param config
    * @return
    */
  def generatePrettyJsonInstanceConfig(config: InstanceConfig): String = {

    implicit val formats: Formats = Serialization.formats(hints = NoTypeHints)
    writePretty(config)
  }

  /**
    *
    * @param json
    * @return
    */
  def generateInstanceConfigFromJson(json: String): InstanceConfig = {
    implicit val formats: Formats = Serialization.formats(hints = NoTypeHints)
    read[InstanceConfig](json)
  }

}


