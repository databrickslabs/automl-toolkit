package com.databricks.labs.automl.executor.config

object RegressorModels extends Enumeration {
  type RegressorModels = Value
  val TreesRegressor, GBTRegressor, LinearRegression, RandomForestRegressor,
  SVM, XGBoostRegressor, LightGBMHuber, LightGBMFair, LightGBMLasso,
  LightGBMRidge, LightGBMPoisson, LightGBMQuantile, LightGBMMape,
  LightGBMTweedie, LightGBMGamma = Value
}

object ClassiferModels extends Enumeration {
  type ClassifierModels = Value
  val TreesClassifier, GBTClassifier, LogisticRegression, MLPC,
  RandomForestClassifier, XGBoostClassifier, LightGBMBinary, LightGBMMulti,
  LightGBMMultiOVA = Value
}

object ModelSelector extends Enumeration {
  type ModelSelector = Value
  val TreesRegressor, TreesClassifier, GBTRegressor, GBTClassifier,
  LinearRegression, LogisticRegression, MLPC, RandomForestRegressor,
  RandomForestClassifier, SVM, XGBoostRegressor, XGBoostClassifier,
  LightGBMBinary, LightGBMMulti, LightGBMMultiOVA, LightGBMHuber, LightGBMFair,
  LightGBMLasso, LightGBMRidge, LightGBMPoisson, LightGBMQuantile, LightGBMMape,
  LightGBMTweedie, LightGBMGamma = Value
}

object FamilyValidator extends Enumeration {
  type FamilyValidator = Value
  val Trees, NonTrees = Value
}

object PredictionType extends Enumeration {
  type PredictionType = Value
  val Regressor, Classifier = Value
}

/**
  *
  * @param labelCol
  * @param featuresCol
  * @param dateTimeConversionType
  * @param fieldsToIgnoreInVector
  * @param scoringMetric
  * @param scoringOptimizationStrategy
  */
case class GenericConfig(var labelCol: String,
                         var featuresCol: String,
                         var dateTimeConversionType: String,
                         var fieldsToIgnoreInVector: Array[String],
                         var scoringMetric: String,
                         var scoringOptimizationStrategy: String)

case class FeatureEngineeringConfig(
  var dataPrepParallelism: Int,
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
  var scalingPNorm: Double,
  var featureImportanceCutoffType: String,
  var featureImportanceCutoffValue: Double,
  var dataReductionFactor: Double,
  var cardinalitySwitch: Boolean,
  var cardinalityType: String,
  var cardinalityLimit: Int,
  var cardinalityPrecision: Double,
  var cardinalityCheckMode: String,
  var filterPrecision: Double,
  var categoricalNAFillMap: Map[String, String],
  var numericNAFillMap: Map[String, AnyVal],
  var characterNABlanketFillValue: String,
  var numericNABlanketFillValue: Double,
  var naFillMode: String,
  var featureInteractionRetentionMode: String,
  var featureInteractionContinuousDiscretizerBucketCount: Int,
  var featureInteractionParallelism: Int,
  var featureInteractionTargetInteractionPercentage: Double
)

case class SwitchConfig(var naFillFlag: Boolean,
                        var varianceFilterFlag: Boolean,
                        var outlierFilterFlag: Boolean,
                        var pearsonFilterFlag: Boolean,
                        var covarianceFilterFlag: Boolean,
                        var oneHotEncodeFlag: Boolean,
                        var scalingFlag: Boolean,
                        var featureInteractionFlag: Boolean,
                        var dataPrepCachingFlag: Boolean,
                        var autoStoppingFlag: Boolean,
                        var pipelineDebugFlag: Boolean)

case class TunerConfig(var tunerAutoStoppingScore: Double,
                       var tunerParallelism: Int,
                       var tunerKFold: Int,
                       var tunerTrainPortion: Double,
                       var tunerTrainSplitMethod: String,
                       var tunerKSampleSyntheticCol: String,
                       var tunerKSampleKGroups: Int,
                       var tunerKSampleKMeansMaxIter: Int,
                       var tunerKSampleKMeansTolerance: Double,
                       var tunerKSampleKMeansDistanceMeasurement: String,
                       var tunerKSampleKMeansSeed: Long,
                       var tunerKSampleKMeansPredictionCol: String,
                       var tunerKSampleLSHHashTables: Int,
                       var tunerKSampleLSHSeed: Long,
                       var tunerKSampleLSHOutputCol: String,
                       var tunerKSampleQuorumCount: Int,
                       var tunerKSampleMinimumVectorCountToMutate: Int,
                       var tunerKSampleVectorMutationMethod: String,
                       var tunerKSampleMutationMode: String,
                       var tunerKSampleMutationValue: Double,
                       var tunerKSampleLabelBalanceMode: String,
                       var tunerKSampleCardinalityThreshold: Int,
                       var tunerKSampleNumericRatio: Double,
                       var tunerKSampleNumericTarget: Int,
                       var tunerTrainSplitChronologicalColumn: String,
                       var tunerTrainSplitChronologicalRandomPercentage: Double,
                       var tunerSeed: Long,
                       var tunerFirstGenerationGenePool: Int,
                       var tunerNumberOfGenerations: Int,
                       var tunerNumberOfParentsToRetain: Int,
                       var tunerNumberOfMutationsPerGeneration: Int,
                       var tunerGeneticMixing: Double,
                       var tunerGenerationalMutationStrategy: String,
                       var tunerFixedMutationValue: Int,
                       var tunerMutationMagnitudeMode: String,
                       var tunerEvolutionStrategy: String,
                       var tunerGeneticMBORegressorType: String,
                       var tunerGeneticMBOCandidateFactor: Int,
                       var tunerContinuousEvolutionImprovementThreshold: Int,
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
                       var tunerInitialGenerationArraySeed: Long,
                       var tunerOutputDfRepartitionScaleFactor: Int)

case class AlgorithmConfig(var stringBoundaries: Map[String, List[String]],
                           var numericBoundaries: Map[String, (Double, Double)])

case class LoggingConfig(var mlFlowLoggingFlag: Boolean,
                         var mlFlowLogArtifactsFlag: Boolean,
                         var mlFlowTrackingURI: String,
                         var mlFlowExperimentName: String,
                         var mlFlowAPIToken: String,
                         var mlFlowModelSaveDirectory: String,
                         var mlFlowLoggingMode: String,
                         var mlFlowBestSuffix: String,
                         var inferenceConfigSaveLocation: String,
                         var mlFlowCustomRunTags: Map[String, String])

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
