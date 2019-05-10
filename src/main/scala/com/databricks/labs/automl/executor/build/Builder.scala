package com.databricks.labs.automl.executor.build

import com.databricks.labs.automl.params.MainConfig


object ModelSelector extends Enumeration {
  type ModelSelector = Value
  val TreesRegressor, TreesClassifier, GBTRegressor, GBTClassifier, LinearRegression, LogisticRegression, MLPC,
  RandomForestRegressor, RandomForestClassifier, SVM, XGBoostRegressor, XGBoostClassifier = Value
}

object FamilyValidator extends Enumeration {
  type FamilyValidator = Value
  val Trees, NonTrees = Value
}

object PredictionType extends Enumeration {
  type PredictionType = Value
  val Regressor, Classifier = Value
}


class GenericConfigGenerator(predictionType: String) extends BuilderDefaults {

  import PredictionType._

  private val familyType: PredictionType = predictionTypeEvaluator(predictionType)

  private var _genericConfig = genericConfig(familyType)


  def setLabelCol(value: String): this.type = {
    _genericConfig.labelCol = value; this
  }

  def setFeaturesCol(value: String): this.type = {
    _genericConfig.featuresCol = value; this
  }

  def setDateTimeConversionType(value: String): this.type = {
    validateMembership(value, allowableDateTimeConversionTypes, "DateTimeConversionType")
    _genericConfig.dateTimeConversionType = value
    this
  }

  def setFieldsToIgnoreInVector(value: Array[String]): this.type = {
    _genericConfig.fieldsToIgnoreInVector = value; this
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

class ConfigurationGenerator(modelFamily: String, predictionType: String, genericConfig: GenericConfig)
  extends BuilderDefaults {

  import FamilyValidator._
  import ModelSelector._

  private val modelType: ModelSelector = modelTypeEvaluator(modelFamily, predictionType)
  private val family: FamilyValidator = familyTypeEvaluator(modelFamily)

  /**
    * Default configuration generation
    */

  //  private var _switchConfig = switchConfig(family)
  //  private var _algorithmConfig = algorithmConfig(modelType)
  //  private var _featureEngineeringConfig = featureEngineeringConfig()


  private var _instanceConfig = instanceConfig(modelFamily, predictionType)

  /**
    * Switch Config
    */

  def naFillOn(): this.type = {
    _instanceConfig.switchConfig.naFillFlag = true; this
  }

  def naFillOff(): this.type = {
    _instanceConfig.switchConfig.naFillFlag = false; this
  }

  def varianceFilterOn(): this.type = {
    _instanceConfig.switchConfig.varianceFilterFlag = true; this
  }

  def varianceFilterOff(): this.type = {
    _instanceConfig.switchConfig.varianceFilterFlag = false; this
  }

  def outlierFilterOn(): this.type = {
    _instanceConfig.switchConfig.outlierFilterFlag = true; this
  }

  def outlierFilterOff(): this.type = {
    _instanceConfig.switchConfig.outlierFilterFlag = false; this
  }

  def pearsonFilterOn(): this.type = {
    _instanceConfig.switchConfig.pearsonFilterFlag = true; this
  }

  def pearsonFilterOff(): this.type = {
    _instanceConfig.switchConfig.pearsonFilterFlag = false; this
  }

  def covarianceFilterOn(): this.type = {
    _instanceConfig.switchConfig.covarianceFilterFlag = true; this
  }

  def covarianceFilterOff(): this.type = {
    _instanceConfig.switchConfig.covarianceFilterFlag = false; this
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
    _instanceConfig.switchConfig.oneHotEncodeFlag = false; this
  }

  def scalingOn(): this.type = {
    _instanceConfig.switchConfig.scalingFlag = true; this
  }

  def scalingOff(): this.type = {
    _instanceConfig.switchConfig.scalingFlag = false; this
  }

  def dataPrepCachingOn(): this.type = {
    _instanceConfig.switchConfig.dataPrepCachingFlag = true; this
  }

  def dataPrepCachingOff(): this.type = {
    _instanceConfig.switchConfig.dataPrepCachingFlag = false; this
  }

  def autoStoppingOn(): this.type = {
    _instanceConfig.switchConfig.autoStoppingFlag = true; this
  }

  def autoStoppingOff(): this.type = {
    _instanceConfig.switchConfig.autoStoppingFlag = false; this
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
    _instanceConfig.featureEngineeringConfig.scalingMin = value; this
  }

  def setScalingMax(value: Double): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingMax = value; this
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
    _instanceConfig.featureEngineeringConfig.scalingStdDevFlag = true; this
  }

  def setScalingStdDevFlagOff(): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingStdDevFlag = false; this
  }

  def setScalingPNorm(value: Double): this.type = {
    require(value >= 1.0, s"pNorm value: $value is invalid. Value must be greater than or equal to 1.0.")
    _instanceConfig.featureEngineeringConfig.scalingPNorm = value
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
    if(value > 30) println("WARNING - Setting Tuner Parallelism greater than 30 could put excessive stress on the " +
      "Driver.  Ensure driver is monitored for stability.")
    _instanceConfig.tunerConfig.tunerParallelism = value
    this
  }
  def setTunerKFold(value: Int): this.type = {
    if(value < 5) println("WARNING - Setting KFold < 5 may result in a poorly generalized tuning run due to " +
      "over-fitting within a particular train/test split.")
    _instanceConfig.tunerConfig.tunerKFold = value
    this
  }
  def setTunerTrainPortion(value: Double): this.type = {
    require(value > 0.0 & value < 1.0, s"TunerTrainPortion must be within the range of 0.0 to 1.0.")
    if(value < 0.5) println(s"WARNING - setting TunerTrainPortion below 0.5 may result in a poorly fit model.  Best" +
      s" practices guidance typically adheres to a 0.7 or 0.8 test/train ratio.")
    _instanceConfig.tunerConfig.tunerTrainPortion = value
    this
  }
  def setTunerTrainSplitMethod(value: String): this.type = {
    validateMembership(value, allowableTrainSplitMethods, "TunerTrainSplitMethod")
    _instanceConfig.tunerConfig.tunerTrainSplitMethod = value
    this
  }

  //TODO: any settings here should mutate and add to the ignored fields for the feature vector!!!!
  def setTunerTrainSplitChronologicalColumn(value: String): this.type = {
    _instanceConfig.tunerConfig.tunerTrainSplitChronologicalColumn = value
    this
  }
  def setTunerTrainSplitChronologicalRandomPercentage(value: Double): this.type = {
    if(value > 10) println("[WARNING] TunerTrainSplitChronologicalRandomPercentage setting this value above 10 " +
    "percent will cause significant per-run train/test skew and variability in row counts during training.  " +
      "Use higher values only if this is desired.")
    _instanceConfig.tunerConfig.tunerTrainSplitChronologicalRandomPercentage = value
    this
  }
  def setTunerSeed(value: Long): this.type = {
    _instanceConfig.tunerConfig.tunerSeed = value; this
  }

  def setTunerFirstGenerationGenePool(value: Int): this.type = ??? //warning
  def setTunerNumberOfGenerations(value: Int): this.type = ??? //warning
  def setTunerNumberOfParentsToRetain(value: Int): this.type = ??? //warning
  def setTunerNumberOfMutationPerGeneration(value: Int): this.type = ??? //warning?
  def setTunerGeneticMixing(value: Double): this.type = ??? //restriction
  def setTunerGenerationalMutationStrategy(value: String): this.type = ??? //restriction
  def setTunerFixedMutationValue(value: Int): this.type = ??? //warning?
  def setTunerMutationMagnitudeMode(value: String): this.type = ??? //restriction
  def setTunerEvolutionStrategy(value: String): this.type = ??? //restriction
  def setTunerContinuousEvolutionMaxIterations(value: Int): this.type = ??? // warning
  def setTunerContinuousEvolutionStoppingScore(value: Double): this.type = ??? // warning + restriction
  def setTunerContinuousEvolutionParallelism(value: Int): this.type = ??? //warning
  def setTunerContinuousEvolutionMutationAggressiveness(value: Int): this.type = ??? // warning
  def setTunerContinuousEvolutionGeneticMixing(value: Double): this.type = ??? //restriction
  def setTunerContinuousEvolutionRollingImprovementCount(value: Int): this.type = ??? //restriction?
  def setTunerModelSeed(value: Map[String, Any]): this.type = ???

  def setTunerHyperSpaceInferenceOn(): this.type = {
    _instanceConfig.tunerConfig.tunerHyperSpaceInference = true; this
  }

  def setTunerHyperSpaceInferenceOff(): this.type = {
    _instanceConfig.tunerConfig.tunerHyperSpaceInference = false; this
  }

  def setTunerHyperSpaceInferenceCount(value: Int): this.type = ??? //warning
  def setTunerHyperSpaceModelCount(value: Int): this.type = ??? //warning
  def setTunerHyperSpaceModelType(value: String): this.type = ??? //restrictions
  def setTunerInitialGenerationMode(value: String): this.type = ??? //restrictions
  def setTunerInitialGenerationPermutationCount(value: Int): this.type = ??? //warning
  def setTunerInitialGenerationIndexMixingMode(value: String): this.type = ??? //restrictions
  def setTunerInitialGenerationArraySeed(value: Long): this.type = {
    _instanceConfig.tunerConfig.tunerInitialGenerationArraySeed = value
    this
  }

  // def zeroToOneValidation()


  /**
    * MLFlow Logging Config
    */

  /**
    * Getters
    */


  def getInstanceConfig: InstanceConfig = _instanceConfig

  //TODO : json input and extract for case class definitions
  //TODO: implicit reflection for map type config?

  def generateMainConfig: MainConfig = ??? //TODO: just build it here.
  def generateFeatureImportanceConfig: MainConfig = ??? //TODO: Needed?
  def generateTreeSplitConfig: MainConfig = ??? //TODO: Needed?

}


object ConfigurationGenerator extends BuilderDefaults {

  import PredictionType._

  def apply(modelFamily: String, predictionType: String, genericConfig: GenericConfig): ConfigurationGenerator =
    new ConfigurationGenerator(modelFamily, predictionType, genericConfig)

  def generateDefaultConfig(modelFamily: String, predictionType: String): InstanceConfig = {

    predictionTypeEvaluator(predictionType) match {
      case Regressor => new ConfigurationGenerator(modelFamily, predictionType,
        GenericConfigGenerator.generateDefaultRegressorConfig).getInstanceConfig
      case Classifier => new ConfigurationGenerator(modelFamily, predictionType,
        GenericConfigGenerator.generateDefaultClassifierConfig).getInstanceConfig
    }

  }

}

case class GenericConfig(
                          var labelCol: String,
                          var featuresCol: String,
                          var dateTimeConversionType: String,
                          var fieldsToIgnoreInVector: Array[String],
                          var scoringMetric: String,
                          var scoringOptimizationStrategy: String
                        )

case class FeatureEngineeringConfig(
                                     var numericFillStat: String,
                                     var characterFillStat: String,
                                     var modelSelectionDistinctThreshold: Int,
                                     var outlierFilterBounds: String,
                                     var outlierLowerFilterNTile: Double,
                                     var outlierUpperFilterNTile: Double,
                                     var outlierFilterPrecision: Double,
                                     var outlierContinuousDataThreshold: Int,
                                     var outlierFieldsToIgnore: Array[String],
                                     var pearsonFilterStatistic: String,
                                     var pearsonFilterDirection: String,
                                     var pearsonFilterManualValue: Double,
                                     var pearsonFilterMode: String,
                                     var pearsonAutoFilterNTile: Double,
                                     var covarianceCorrelationCutoffLow: Double,
                                     var covarianceCorrelationCutoffHigh: Double,
                                     var scalingType: String,
                                     var scalingMin: Double,
                                     var scalingMax: Double,
                                     var scalingStandardMeanFlag: Boolean,
                                     var scalingStdDevFlag: Boolean,
                                     var scalingPNorm: Double
                                   )

case class SwitchConfig(
                         var naFillFlag: Boolean,
                         var varianceFilterFlag: Boolean,
                         var outlierFilterFlag: Boolean,
                         var pearsonFilterFlag: Boolean,
                         var covarianceFilterFlag: Boolean,
                         var oneHotEncodeFlag: Boolean,
                         var scalingFlag: Boolean,
                         var dataPrepCachingFlag: Boolean,
                         var autoStoppingFlag: Boolean
                       )


case class TunerConfig(
                        var tunerAutoStoppingScore: Double,
                        var tunerParallelism: Int,
                        var tunerKFold: Int,
                        var tunerTrainPortion: Double,
                        var tunerTrainSplitMethod: String,
                        var tunerTrainSplitChronologicalColumn: String,
                        var tunerTrainSplitChronologicalRandomPercentage: Double,
                        var tunerSeed: Long,
                        var tunerFirstGenerationGenePool: Int,
                        var tunerNumberOfGenerations: Int,
                        var tunerNumerOfParentsToRetain: Int,
                        var tunerNumberOfMutationsPerGeneration: Int,
                        var tunerGeneticMixing: Double,
                        var tunerGenerationalMutationStrategy: String,
                        var tunerFixedMutationValue: Int,
                        var tunerMutationMagnitudeMode: String,
                        var tunerEvolutionStrategy: String,
                        var tunerContinuousEvolutionMaxIterations: Int,
                        var tunerContinuousEvolutionStoppingScore: Double,
                        var tunerContinuousEvolutionParallelism: Int,
                        var tunerContinuousEvolutionMutationAggressiveness: Int,
                        var tunerContinuousEvolutionGeneticMixing: Double,
                        var tunerContinuousEvolutionRollingImprovingCount: Int,
                        var tunerModelSeed: Map[String, Any],
                        var tunerHyperSpaceInference: Boolean,
                        var tunerHyperSpaceInferenceCount: Int,
                        var tunerHyperSpaceModelCount: Int,
                        var tunerHyperSpaceModelType: String,
                        var tunerInitialGenerationMode: String,
                        var tunerInitialGenerationPermutationCount: Int,
                        var tunerInitialGenerationIndexMixingMode: String,
                        var tunerInitialGenerationArraySeed: Long
                      )

case class AlgorithmConfig(
                            var stringBoundaries: Map[String, List[String]],
                            var numericBoundaries: Map[String, (Double, Double)]
                          )

case class LoggingConfig(

                        )


case class InstanceConfig(
                           var modelFamily: String,
                           var predictionType: String,
                           var genericConfig: GenericConfig,
                           var switchConfig: SwitchConfig,
                           var featureEngineeringConfig: FeatureEngineeringConfig,
                           var algorithmConfig: AlgorithmConfig,
                           var tunerConfig: TunerConfig,
                           var loggingConfig: LoggingConfig
                         )



