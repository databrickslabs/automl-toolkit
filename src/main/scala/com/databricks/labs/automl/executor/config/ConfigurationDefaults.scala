package com.databricks.labs.automl.executor.config

import com.databricks.labs.automl.utils.InitDbUtils

trait ConfigurationDefaults {

  import FamilyValidator._
  import ModelDefaults._
  import ModelSelector._
  import PredictionType._

  /**
    * General Tools
    */
  private[config] def modelTypeEvaluator(
    modelFamily: String,
    predictionType: String
  ): ModelSelector = {
    (
      modelFamily.toLowerCase.replaceAll("\\s", ""),
      predictionType.toLowerCase.replaceAll("\\s", "")
    ) match {
      case ("trees", "regressor")               => TreesRegressor
      case ("trees", "classifier")              => TreesClassifier
      case ("gbt", "regressor")                 => GBTRegressor
      case ("gbt", "classifier")                => GBTClassifier
      case ("randomforest", "regressor")        => RandomForestRegressor
      case ("randomforest", "classifier")       => RandomForestClassifier
      case ("linearregression", "regressor")    => LinearRegression
      case ("logisticregression", "classifier") => LogisticRegression
      case ("xgboost", "regressor")             => XGBoostRegressor
      case ("xgboost", "classifier")            => XGBoostClassifier
      case ("mlpc", "classifier")               => MLPC
      case ("svm", "classifier")                => SVM
//      case ("gbmbinary", "classifier")          => LightGBMBinary // turning these off until LightGBM is fixed by MSFT
//      case ("gbmmulti", "classifier")           => LightGBMMulti
//      case ("gbmmultiova", "classifier")        => LightGBMMultiOVA
//      case ("gbmhuber", "regressor")            => LightGBMHuber
//      case ("gbmfair", "regressor")             => LightGBMFair
//      case ("gbmlasso", "regressor")            => LightGBMLasso
//      case ("gbmridge", "regressor")            => LightGBMRidge
//      case ("gbmpoisson", "regressor")          => LightGBMPoisson
//      case ("gbmquantile", "regressor")         => LightGBMQuantile
//      case ("gbmmape", "regressor")             => LightGBMMape
//      case ("gbmtweedie", "regressor")          => LightGBMTweedie
//      case ("gbmgamma", "regressor")            => LightGBMGamma
      case (_, _) =>
        throw new IllegalArgumentException(
          s"'$modelFamily' Model Family and PredictionType " +
            s"'$predictionType' are not supported."
        )
    }
  }

  private[config] def predictionTypeEvaluator(
    predictionType: String
  ): PredictionType = {
    predictionType.toLowerCase.replaceAll("\\s", "") match {
      case "regressor"  => Regressor
      case "classifier" => Classifier
      case _ =>
        throw new IllegalArgumentException(
          s"'$predictionType' is not a supported type! Must be either: " +
            s"'regressor' or 'classifier'"
        )
    }
  }

  private[config] def familyTypeEvaluator(
    modelFamily: String
  ): FamilyValidator = {
    modelFamily.toLowerCase.replaceAll("\\s", "") match {
      case "trees" | "gbt" | "randomforest" | "xgboost" => Trees
      case _                                            => NonTrees
    }
  }

  private[config] def zeroToOneValidation(value: Double,
                                          parameterName: String): Unit = {
    require(
      value >= 0.0 & value <= 1.0,
      s"$parameterName submitted value of '$value' is outside of the allowable " +
        s"bounds of 0.0 to 1.0."
    )
  }

  private[config] def validateMembership(value: String,
                                         collection: List[String],
                                         parameterName: String): Unit = {
    require(
      collection.contains(value),
      s"$parameterName value '$value' is not supported.  Must be one of: '" +
        s"${collection.mkString(", ")}'"
    )
  }

  /**
    * Static restrictions
    */
  final val allowableDateTimeConversionTypes: List[String] =
    List("unix", "split")
  final val allowableRegressionScoringMetrics: List[String] =
    List("rmse", "mse", "r2", "mae")
  final val allowableClassificationScoringMetrics: List[String] = List(
    "f1",
    "weightedPrecision",
    "weightedRecall",
    "accuracy",
    "areaUnderPR",
    "areaUnderROC"
  )
  final val allowableScoringOptimizationStrategies: List[String] =
    List("minimize", "maximize")
  final val allowableNumericFillStats: List[String] =
    List("min", "25p", "mean", "median", "75p", "max")
  final val allowableCharacterFillStats: List[String] = List("min", "max")
  final val allowableOutlierFilterBounds: List[String] =
    List("lower", "upper", "both")
  final val allowablePearsonFilterStats: List[String] =
    List("pvalue", "degreesFreedom", "pearsonStat")
  final val allowablePearsonFilterDirections: List[String] =
    List("greater", "lesser")
  final val allowablePearsonFilterModes: List[String] = List("auto", "manual")
  final val allowableScalers: List[String] =
    List("minMax", "standard", "normalize", "maxAbs")
  final val allowableTrainSplitMethods: List[String] = List(
    "random",
    "chronological",
    "stratifyReduce",
    "stratified",
    "overSample",
    "underSample",
    "kSample"
  )
  final val allowableEvolutionStrategies: List[String] =
    List("batch", "continuous")
  final val allowableMlFlowLoggingModes: List[String] =
    List("tuningOnly", "bestOnly", "full")
  final val allowableInitialGenerationModes: List[String] =
    List("random", "permutations")
  final val allowableInitialGenerationIndexMixingModes: List[String] =
    List("random", "linear")
  final val allowableMutationStrategies: List[String] = List("linear", "fixed")
  final val allowableMutationMagnitudeMode: List[String] =
    List("random", "fixed")
  final val allowableHyperSpaceModelTypes: List[String] =
    List("RandomForest", "LinearRegression", "XGBoost")
  final val allowableFeatureImportanceCutoffTypes: List[String] =
    List("none", "value", "count")
  final val allowableKMeansDistanceMeasurements: List[String] =
    List("cosine", "euclidean")
  final val allowableMutationModes: List[String] =
    List("weighted", "random", "ratio")
  final val allowableVectorMutationMethods: List[String] =
    List("random", "fixed", "all")
  final val allowableLabelBalanceModes: List[String] =
    List("match", "percentage", "target")
  final val allowableDateTimeConversions: List[String] = List("unix", "split")
  final val allowableCategoricalFilterModes: List[String] =
    List("silent", "warn")
  final val allowableCardinalilties: List[String] = List("approx", "exact")
  final val allowableNAFillModes: List[String] =
    List(
      "auto",
      "mapFill",
      "blanketFillAll",
      "blanketFillCharOnly",
      "blanketFillNumOnly"
    )
  final val allowableGeneticMBORegressorTypes: List[String] =
    List("XGBoost", "LinearRegression", "RandomForest")

  final val allowableFeatureInteractionModes =
    List("optimistic", "strict", "all")

  /**
    * Generic Helper Methods
    */
  private def familyScoringCheck(predictionType: PredictionType): String = {
    predictionType match {
      case Regressor => "rmse"
      case _         => "areaUnderROC"
    }
  }

  private def familyScoringCheck(predictionType: String): String = {
    familyScoringCheck(predictionTypeEvaluator(predictionType))
  }

  private def treesBooleanSwitch(modelType: FamilyValidator): Boolean = {
    modelType match {
      case Trees => false
      case _     => true
    }
  }

  def oneHotEncodeFlag(family: FamilyValidator): Boolean =
    treesBooleanSwitch(family)
  def scalingFlag(family: FamilyValidator): Boolean = treesBooleanSwitch(family)

  private def familyScoringDirection(predictionType: PredictionType): String = {
    predictionType match {
      case Regressor => "minimize"
      case _         => "maximize"
    }
  }

  private def familyScoringDirection(predictionType: String): String = {
    familyScoringDirection(predictionTypeEvaluator(predictionType))
  }

  /**
    * Algorithm Helper Methods
    */
  private[config] def boundaryValidation(modelKeys: Set[String],
                                         overwriteKeys: Set[String]): Unit = {
    require(
      modelKeys == overwriteKeys,
      s"The provided configuration does not match. Expected: " +
        s"${modelKeys.mkString(", ")}, but got: ${overwriteKeys.mkString(", ")} }"
    )
  }

  private[config] def validateNumericBoundariesKeys(
    modelType: ModelSelector,
    value: Map[String, (Double, Double)]
  ): Unit = {
    modelType match {
      case RandomForestRegressor =>
        boundaryValidation(randomForestNumeric.keys.toSet, value.keys.toSet)
      case RandomForestClassifier =>
        boundaryValidation(randomForestNumeric.keys.toSet, value.keys.toSet)
      case TreesRegressor =>
        boundaryValidation(treesNumeric.keys.toSet, value.keys.toSet)
      case TreesClassifier =>
        boundaryValidation(treesNumeric.keys.toSet, value.keys.toSet)
      case XGBoostRegressor =>
        boundaryValidation(xgBoostNumeric.keys.toSet, value.keys.toSet)
      case XGBoostClassifier =>
        boundaryValidation(xgBoostNumeric.keys.toSet, value.keys.toSet)
      case MLPC => boundaryValidation(mlpcNumeric.keys.toSet, value.keys.toSet)
      case GBTRegressor =>
        boundaryValidation(gbtNumeric.keys.toSet, value.keys.toSet)
      case GBTClassifier =>
        boundaryValidation(gbtNumeric.keys.toSet, value.keys.toSet)
      case LinearRegression =>
        boundaryValidation(linearRegressionNumeric.keys.toSet, value.keys.toSet)
      case LogisticRegression =>
        boundaryValidation(
          logisticRegressionNumeric.keys.toSet,
          value.keys.toSet
        )
      case SVM => boundaryValidation(svmNumeric.keys.toSet, value.keys.toSet)
      case LightGBMBinary | LightGBMMulti | LightGBMMultiOVA | LightGBMHuber |
          LightGBMFair | LightGBMLasso | LightGBMLasso | LightGBMRidge |
          LightGBMPoisson | LightGBMQuantile | LightGBMMape | LightGBMTweedie |
          LightGBMGamma =>
        boundaryValidation(lightGBMnumeric.keys.toSet, value.keys.toSet)
    }
  }

  private[config] def validateNumericBoundariesValues(
    values: Map[String, (Double, Double)]
  ): Unit = {
    values.foreach(
      k =>
        require(
          k._2._1 < k._2._2,
          s"Numeric Boundary key ${k._1} is set incorrectly! " +
            s"Boundary definitions must be in the form: (min, max)"
      )
    )
  }

  private[config] def numericBoundariesAssignment(
    modelType: ModelSelector
  ): Map[String, (Double, Double)] = {
    modelType match {
      case RandomForestRegressor  => randomForestNumeric
      case RandomForestClassifier => randomForestNumeric
      case TreesRegressor         => treesNumeric
      case TreesClassifier        => treesNumeric
      case XGBoostRegressor       => xgBoostNumeric
      case XGBoostClassifier      => xgBoostNumeric
      case MLPC                   => mlpcNumeric
      case GBTRegressor           => gbtNumeric
      case GBTClassifier          => gbtNumeric
      case LinearRegression       => linearRegressionNumeric
      case LogisticRegression     => logisticRegressionNumeric
      case SVM                    => svmNumeric
      case LightGBMBinary | LightGBMMulti | LightGBMMultiOVA | LightGBMHuber |
          LightGBMFair | LightGBMLasso | LightGBMLasso | LightGBMRidge |
          LightGBMPoisson | LightGBMQuantile | LightGBMMape | LightGBMTweedie |
          LightGBMGamma =>
        lightGBMnumeric
      case _ =>
        throw new NotImplementedError(
          s"Model Type ${modelType.toString} is not implemented."
        )
    }
  }

  private[config] def validateStringBoundariesKeys(
    modelType: ModelSelector,
    value: Map[String, List[String]]
  ): Unit = {
    modelType match {
      case RandomForestRegressor =>
        boundaryValidation(randomForestString.keys.toSet, value.keys.toSet)
      case RandomForestClassifier =>
        boundaryValidation(randomForestString.keys.toSet, value.keys.toSet)
      case TreesRegressor =>
        boundaryValidation(treesString.keys.toSet, value.keys.toSet)
      case TreesClassifier =>
        boundaryValidation(treesString.keys.toSet, value.keys.toSet)
      case MLPC => boundaryValidation(mlpcString.keys.toSet, value.keys.toSet)
      case GBTRegressor =>
        boundaryValidation(gbtString.keys.toSet, value.keys.toSet)
      case GBTClassifier =>
        boundaryValidation(gbtString.keys.toSet, value.keys.toSet)
      case LinearRegression =>
        boundaryValidation(linearRegressionString.keys.toSet, value.keys.toSet)
      case LightGBMBinary | LightGBMMulti | LightGBMMultiOVA | LightGBMHuber |
          LightGBMFair | LightGBMLasso | LightGBMLasso | LightGBMRidge |
          LightGBMPoisson | LightGBMQuantile | LightGBMMape | LightGBMTweedie |
          LightGBMGamma =>
        boundaryValidation(lightGBMString.keys.toSet, value.keys.toSet)
      case _ => None
    }
  }

  private[config] def stringBoundariesAssignment(
    modelType: ModelSelector
  ): Map[String, List[String]] = {
    modelType match {
      case RandomForestRegressor  => randomForestString
      case RandomForestClassifier => randomForestString
      case TreesRegressor         => treesString
      case TreesClassifier        => treesString
      case XGBoostRegressor       => Map.empty
      case XGBoostClassifier      => Map.empty
      case MLPC                   => mlpcString
      case GBTRegressor           => gbtString
      case GBTClassifier          => gbtString
      case LinearRegression       => linearRegressionString
      case LogisticRegression     => Map.empty
      case SVM                    => Map.empty
      case LightGBMBinary | LightGBMMulti | LightGBMMultiOVA | LightGBMHuber |
          LightGBMFair | LightGBMLasso | LightGBMLasso | LightGBMRidge |
          LightGBMPoisson | LightGBMQuantile | LightGBMMape | LightGBMTweedie |
          LightGBMGamma =>
        lightGBMString
      case _ =>
        throw new NotImplementedError(
          s"Model Type ${modelType.toString} is not implemented."
        )
    }
  }

  /**
    * Generate the default configuration objects
    */
  private[config] def genericConfig(
    predictionType: PredictionType
  ): GenericConfig = {
    val labelCol = "label"
    val featuresCol = "features"
    val dateTimeConversionType = "split"
    val fieldsToIgnoreInVector = Array.empty[String]
    val scoringMetric = familyScoringCheck(predictionType)
    val scoringOptimizationStrategy = familyScoringDirection(predictionType)

    GenericConfig(
      labelCol,
      featuresCol,
      dateTimeConversionType,
      fieldsToIgnoreInVector,
      scoringMetric,
      scoringOptimizationStrategy
    )
  }

  private[config] def switchConfig(family: FamilyValidator): SwitchConfig = {
    val naFillFlag = true
    val varianceFilterFlag = true
    val outlierFilterFlag = false
    val pearsonFilterFlag = false
    val covarianceFilterFlag = false
    val oheFlag = oneHotEncodeFlag(family)
    val scaleFlag = scalingFlag(family)
    val dataPrepCachingFlag = true
    val autoStoppingFlag = false
    val pipelineDebugFlag = false
    val featureInteractionFlag = false

    SwitchConfig(
      naFillFlag,
      varianceFilterFlag,
      outlierFilterFlag,
      pearsonFilterFlag,
      covarianceFilterFlag,
      oheFlag,
      scaleFlag,
      featureInteractionFlag,
      dataPrepCachingFlag,
      autoStoppingFlag,
      pipelineDebugFlag
    )
  }

  private[config] def algorithmConfig(
    modelType: ModelSelector
  ): AlgorithmConfig =
    AlgorithmConfig(
      stringBoundariesAssignment(modelType),
      numericBoundariesAssignment(modelType)
    )

  private[config] def featureEngineeringConfig(): FeatureEngineeringConfig = {
    val dataPrepParallelism = 20
    val numericFillStat = "mean"
    val characterFillStat = "max"
    val modelSelectionDistinctThreshold = 50
    val outlierFilterBounds = "both"
    val outlierLowerFilterNTile = 0.02
    val outlierUpperFilterNTile = 0.98
    val outlierFilterPrecision = 0.01
    val outlierContinuousDataThreshold = 50
    val outlierFieldsToIgnore = Array.empty[String]
    val pearsonFilterStatistic = "pValue"
    val pearsonFilterDirection = "greater"
    val pearsonFilterManualValue = 0.0
    val pearsonFilterMode = "auto"
    val pearsonAutoFilterNTile = 0.75
    val covarianceCorrelationCutoffLow = -0.8
    val covarianceCorrelctionCutoffHigh = 0.8
    val scalingType = "minMax"
    val scalingMin = 0.0
    val scalingMax = 1.0
    val scalingStandardMeanFlag = false
    val scalingStdDevFlag = true
    val scalingPNorm = 2.0
    val featureImportanceCutoffType = "count"
    val featureImportanceCutoffValue = 15.0
    val dataReductionFactor = 0.5
    val cardinalitySwitch = true
    val cardinalityType = "exact"
    val cardinalityLimit = 200
    val cardinalityPrecision = 0.05
    val cardinalityCheckMode = "silent"
    val filterPrecision = 0.01
    val categoricalNAFillMap = Map.empty[String, String]
    val numericNAFillMap = Map.empty[String, AnyVal]
    val characterNABlanketFillValue = ""
    val numericNABlanketFillValue = 0.0
    val naFillMode = "auto"
    val featureInteractionRetentionMode = "optimistic"
    val featureInteractionContinuousDiscretizerBucketCount = 10
    val featureInteractionParallelism = 12
    val featureInteractionTargetInteractionPercentage = 10.0

    FeatureEngineeringConfig(
      dataPrepParallelism,
      numericFillStat,
      characterFillStat,
      modelSelectionDistinctThreshold,
      outlierFilterBounds,
      outlierLowerFilterNTile,
      outlierUpperFilterNTile,
      outlierFilterPrecision,
      outlierContinuousDataThreshold,
      outlierFieldsToIgnore,
      pearsonFilterStatistic,
      pearsonFilterDirection,
      pearsonFilterManualValue,
      pearsonFilterMode,
      pearsonAutoFilterNTile,
      covarianceCorrelationCutoffLow,
      covarianceCorrelctionCutoffHigh,
      scalingType,
      scalingMin,
      scalingMax,
      scalingStandardMeanFlag,
      scalingStdDevFlag,
      scalingPNorm,
      featureImportanceCutoffType,
      featureImportanceCutoffValue,
      dataReductionFactor,
      cardinalitySwitch,
      cardinalityType,
      cardinalityLimit,
      cardinalityPrecision,
      cardinalityCheckMode,
      filterPrecision,
      categoricalNAFillMap,
      numericNAFillMap,
      characterNABlanketFillValue,
      numericNABlanketFillValue,
      naFillMode,
      featureInteractionRetentionMode,
      featureInteractionContinuousDiscretizerBucketCount,
      featureInteractionParallelism,
      featureInteractionTargetInteractionPercentage
    )
  }

  private[config] def tunerConfig(): TunerConfig = {
    val tunerAutoStoppingScore = 0.99
    val tunerParallelism = 20
    val tunerKFold = 5
    val tunerTrainPortion = 0.8
    val tunerTrainSplitMethod = "random"
    val tunerKSampleSyntheticCol = "synthetic_ksample"
    val tunerKSampleKGroups = 25
    val tunerKSampleKMeansMaxIter = 100
    val tunerKSampleKMeansTolerance = 1E-6
    val tunerKSampleKMeansDistanceMeasurement = "euclidean"
    val tunerKSampleKMeansSeed = 42L
    val tunerKSampleKMeansPredictionCol = "kGroups_ksample"
    val tunerKSampleLSHHashTables = 10
    val tunerKSampleLSHSeed = 42L
    val tunerKSampleLSHOutputCol = "hashes_ksample"
    val tunerKSampleQuorumCount = 7
    val tunerKSampleMinimumVectorCountToMutate = 1
    val tunerKSampleVectorMutationMethod = "random"
    val tunerKSampleMutationMode = "weighted"
    val tunerKSampleMutationValue = 0.5
    val tunerKSampleLabelBalanceMode = "match"
    val tunerKSampleCardinalityThreshold = 20
    val tunerKSampleNumericRatio = 0.2
    val tunerKSampleNumericTarget = 500
    val tunerTrainSplitChronologicalColumn = ""
    val tunerTrainSplitChronologicalRandomPercentage = 0.0
    val tunerSeed = 42L
    val tunerFirstGenerationGenePool = 20
    val tunerNumberOfGenerations = 10
    val tunerNumberOfParentsToRetain = 3
    val tunerNumberOfMutationsPerGeneration = 10
    val tunerGeneticMixing = 0.7
    val tunerGenerationMutationStrategy = "linear"
    val tunerFixedMutationValue = 1
    val tunerMutationMagnitudeMode = "fixed"
    val tunerEvolutionStrategy = "batch"
    val tunerGeneticMBORegressorType = "XGBoost"
    val tunerGeneticMBOCandidateFactor = 10
    val tunerContinuousImprovementThreshold = -10
    val tunerContinuousEvolutionMaxIterations = 200
    val tunerContinuousEvolutionStoppingScore = 1.0
    val tunerContinuousEvolutionParallelism = 4
    val tunerContinuousEvolutionMutationAggressiveness = 3
    val tunerContinuousEvolutionGeneticMixing = 0.7
    val tunerContinuousEvolutionRollingImprovementCount = 20
    val tunerModelSeed = Map.empty[String, Any]
    val tunerHyperSpaceInference = true
    val tunerHyperSpaceInferenceCount = 200000
    val tunerHyperSpaceModelCount = 10
    val tunerHyperSpaceModelType = "RandomForest"
    val tunerInitialGenerationMode = "random"
    val tunerInitialGenerationPermutationCount = 10
    val tunerInitialGenerationIndexMixingMode = "linear"
    val tunerInitialGenerationArraySeed = 42L
    val tunerOutputDfRepartitionScaleFactor = 3

    TunerConfig(
      tunerAutoStoppingScore,
      tunerParallelism,
      tunerKFold,
      tunerTrainPortion,
      tunerTrainSplitMethod,
      tunerKSampleSyntheticCol,
      tunerKSampleKGroups,
      tunerKSampleKMeansMaxIter,
      tunerKSampleKMeansTolerance,
      tunerKSampleKMeansDistanceMeasurement,
      tunerKSampleKMeansSeed,
      tunerKSampleKMeansPredictionCol,
      tunerKSampleLSHHashTables,
      tunerKSampleLSHSeed,
      tunerKSampleLSHOutputCol,
      tunerKSampleQuorumCount,
      tunerKSampleMinimumVectorCountToMutate,
      tunerKSampleVectorMutationMethod,
      tunerKSampleMutationMode,
      tunerKSampleMutationValue,
      tunerKSampleLabelBalanceMode,
      tunerKSampleCardinalityThreshold,
      tunerKSampleNumericRatio,
      tunerKSampleNumericTarget,
      tunerTrainSplitChronologicalColumn,
      tunerTrainSplitChronologicalRandomPercentage,
      tunerSeed,
      tunerFirstGenerationGenePool,
      tunerNumberOfGenerations,
      tunerNumberOfParentsToRetain,
      tunerNumberOfMutationsPerGeneration,
      tunerGeneticMixing,
      tunerGenerationMutationStrategy,
      tunerFixedMutationValue,
      tunerMutationMagnitudeMode,
      tunerEvolutionStrategy,
      tunerGeneticMBORegressorType,
      tunerGeneticMBOCandidateFactor,
      tunerContinuousImprovementThreshold,
      tunerContinuousEvolutionMaxIterations,
      tunerContinuousEvolutionStoppingScore,
      tunerContinuousEvolutionParallelism,
      tunerContinuousEvolutionMutationAggressiveness,
      tunerContinuousEvolutionGeneticMixing,
      tunerContinuousEvolutionRollingImprovementCount,
      tunerModelSeed,
      tunerHyperSpaceInference,
      tunerHyperSpaceInferenceCount,
      tunerHyperSpaceModelCount,
      tunerHyperSpaceModelType,
      tunerInitialGenerationMode,
      tunerInitialGenerationPermutationCount,
      tunerInitialGenerationIndexMixingMode,
      tunerInitialGenerationArraySeed,
      tunerOutputDfRepartitionScaleFactor
    )
  }

  private[config] def loggingConfig(): LoggingConfig = {
    val mlFlowLoggingFlag = true
    val mlFlowLogArtifactsFlag = false
    val mlfloWLoggingConfig =
      InitDbUtils.getMlFlowLoggingConfig(mlFlowLoggingFlag)
    val mlFlowLoggingMode = "full"
    val mlFlowBestSuffix = "_best"
    val inferenceSaveLocation = "/inference/"
    val mlFlowCustomRunTags = Map[String, String]()

    LoggingConfig(
      mlFlowLoggingFlag,
      mlFlowLogArtifactsFlag,
      mlfloWLoggingConfig.mlFlowTrackingURI,
      mlfloWLoggingConfig.mlFlowExperimentName,
      mlfloWLoggingConfig.mlFlowAPIToken,
      mlfloWLoggingConfig.mlFlowModelSaveDirectory,
      mlFlowLoggingMode,
      mlFlowBestSuffix,
      inferenceSaveLocation,
      mlFlowCustomRunTags
    )
  }

  private[config] def instanceConfig(modelFamily: String,
                                     predictionType: String): InstanceConfig = {
    val modelingType = predictionTypeEvaluator(predictionType)
    val family = familyTypeEvaluator(modelFamily)
    val modelType = modelTypeEvaluator(modelFamily, predictionType)
    InstanceConfig(
      modelFamily,
      predictionType,
      genericConfig(modelingType),
      switchConfig(family),
      featureEngineeringConfig(),
      algorithmConfig(modelType),
      tunerConfig(),
      loggingConfig()
    )
  }

  private[config] def defaultConfigMap(
    modelFamily: String,
    predictionType: String
  ): Map[String, Any] = {

    val genDef = genericConfig(predictionTypeEvaluator(predictionType))
    val switchDef = switchConfig(familyTypeEvaluator(modelFamily))
    val featDef = featureEngineeringConfig()
    val algDef = algorithmConfig(
      modelTypeEvaluator(modelFamily, predictionType)
    )
    val tunerDef = tunerConfig()

    val logDef = loggingConfig()

    Map(
      "labelCol" -> genDef.labelCol,
      "featuresCol" -> genDef.featuresCol,
      "dateTimeConversionType" -> genDef.dateTimeConversionType,
      "fieldsToIgnoreInVector" -> genDef.fieldsToIgnoreInVector,
      "scoringMetric" -> genDef.scoringMetric,
      "scoringOptimizationStrategy" -> genDef.scoringOptimizationStrategy,
      "dataPrepParallelism" -> featDef.dataPrepParallelism,
      "naFillFlag" -> switchDef.naFillFlag,
      "varianceFilterFlag" -> switchDef.varianceFilterFlag,
      "outlierFilterFlag" -> switchDef.outlierFilterFlag,
      "pearsonFilterFlag" -> switchDef.pearsonFilterFlag,
      "covarianceFilterFlag" -> switchDef.covarianceFilterFlag,
      "oneHotEncodeFlag" -> switchDef.oneHotEncodeFlag,
      "scalingFlag" -> switchDef.scalingFlag,
      "featureInteractionFlag" -> switchDef.featureInteractionFlag,
      "dataPrepCachingFlag" -> switchDef.dataPrepCachingFlag,
      "autoStoppingFlag" -> switchDef.autoStoppingFlag,
      "pipelineDebugFlag" -> switchDef.pipelineDebugFlag,
      "fillConfigNumericFillStat" -> featDef.numericFillStat,
      "fillConfigCharacterFillStat" -> featDef.characterFillStat,
      "fillConfigModelSelectionDistinctThreshold" -> featDef.modelSelectionDistinctThreshold,
      "outlierFilterBounds" -> featDef.outlierFilterBounds,
      "outlierLowerFilterNTile" -> featDef.outlierLowerFilterNTile,
      "outlierUpperFilterNTile" -> featDef.outlierUpperFilterNTile,
      "outlierFilterPrecision" -> featDef.outlierFilterPrecision,
      "outlierContinuousDataThreshold" -> featDef.outlierContinuousDataThreshold,
      "outlierFieldsToIgnore" -> featDef.outlierFieldsToIgnore,
      "pearsonFilterStatistic" -> featDef.pearsonFilterStatistic,
      "pearsonFilterDirection" -> featDef.pearsonFilterDirection,
      "pearsonFilterManualValue" -> featDef.pearsonFilterManualValue,
      "pearsonFilterMode" -> featDef.pearsonFilterMode,
      "pearsonAutoFilterNTile" -> featDef.pearsonAutoFilterNTile,
      "covarianceCutoffLow" -> featDef.covarianceCorrelationCutoffLow,
      "covarianceCutoffHigh" -> featDef.covarianceCorrelationCutoffHigh,
      "scalingType" -> featDef.scalingType,
      "scalingMin" -> featDef.scalingMin,
      "scalingMax" -> featDef.scalingMax,
      "scalingStandardMeanFlag" -> featDef.scalingStandardMeanFlag,
      "scalingStdDevFlag" -> featDef.scalingStdDevFlag,
      "scalingPNorm" -> featDef.scalingPNorm,
      "featureInteractionRetentionMode" -> featDef.featureInteractionRetentionMode,
      "featureInteractionContinuousDiscretizerBucketCount" -> featDef.featureInteractionContinuousDiscretizerBucketCount,
      "featureInteractionParallelism" -> featDef.featureInteractionParallelism,
      "featureInteractionTargetInteractionPercentage" -> featDef.featureInteractionTargetInteractionPercentage,
      "featureImportanceCutoffType" -> featDef.featureImportanceCutoffType,
      "featureImportanceCutoffValue" -> featDef.featureImportanceCutoffValue,
      "dataReductionFactor" -> featDef.dataReductionFactor,
      "fillConfigCardinalitySwitch" -> featDef.cardinalitySwitch,
      "fillConfigCardinalityType" -> featDef.cardinalityType,
      "fillConfigCardinalityLimit" -> featDef.cardinalityLimit,
      "fillConfigCardinalityPrecision" -> featDef.cardinalityPrecision,
      "fillConfigCardinalityCheckMode" -> featDef.cardinalityCheckMode,
      "fillConfigFilterPrecision" -> featDef.filterPrecision,
      "fillConfigCategoricalNAFillMap" -> featDef.categoricalNAFillMap,
      "fillConfigNumericNAFillMap" -> featDef.numericNAFillMap,
      "fillConfigCharacterNABlanketFillValue" -> featDef.characterNABlanketFillValue,
      "fillConfigNumericNABlanketFillValue" -> featDef.numericNABlanketFillValue,
      "fillConfigNAFillMode" -> featDef.naFillMode,
      "stringBoundaries" -> algDef.stringBoundaries,
      "numericBoundaries" -> algDef.numericBoundaries,
      "tunerAutoStoppingScore" -> tunerDef.tunerAutoStoppingScore,
      "tunerParallelism" -> tunerDef.tunerParallelism,
      "tunerKFold" -> tunerDef.tunerKFold,
      "tunerTrainPortion" -> tunerDef.tunerTrainPortion,
      "tunerTrainSplitMethod" -> tunerDef.tunerTrainSplitMethod,
      "tunerKSampleSyntheticCol" -> tunerDef.tunerKSampleSyntheticCol,
      "tunerKSampleKGroups" -> tunerDef.tunerKSampleKGroups,
      "tunerKSampleKMeansMaxIter" -> tunerDef.tunerKSampleKMeansMaxIter,
      "tunerKSampleKMeansTolerance" -> tunerDef.tunerKSampleKMeansTolerance,
      "tunerKSampleKMeansDistanceMeasurement" -> tunerDef.tunerKSampleKMeansDistanceMeasurement,
      "tunerKSampleKMeansSeed" -> tunerDef.tunerKSampleKMeansSeed,
      "tunerKSampleKMeansPredictionCol" -> tunerDef.tunerKSampleKMeansPredictionCol,
      "tunerKSampleLSHHashTables" -> tunerDef.tunerKSampleLSHHashTables,
      "tunerKSampleLSHSeed" -> tunerDef.tunerKSampleLSHSeed,
      "tunerKSampleLSHOutputCol" -> tunerDef.tunerKSampleLSHOutputCol,
      "tunerKSampleQuorumCount" -> tunerDef.tunerKSampleQuorumCount,
      "tunerKSampleMinimumVectorCountToMutate" -> tunerDef.tunerKSampleMinimumVectorCountToMutate,
      "tunerKSampleVectorMutationMethod" -> tunerDef.tunerKSampleVectorMutationMethod,
      "tunerKSampleMutationMode" -> tunerDef.tunerKSampleMutationMode,
      "tunerKSampleMutationValue" -> tunerDef.tunerKSampleMutationValue,
      "tunerKSampleLabelBalanceMode" -> tunerDef.tunerKSampleLabelBalanceMode,
      "tunerKSampleCardinalityThreshold" -> tunerDef.tunerKSampleCardinalityThreshold,
      "tunerKSampleNumericRatio" -> tunerDef.tunerKSampleNumericRatio,
      "tunerKSampleNumericTarget" -> tunerDef.tunerKSampleNumericTarget,
      "tunerTrainSplitChronologicalColumn" -> tunerDef.tunerTrainSplitChronologicalColumn,
      "tunerTrainSplitChronologicalRandomPercentage" -> tunerDef.tunerTrainSplitChronologicalRandomPercentage,
      "tunerSeed" -> tunerDef.tunerSeed,
      "tunerFirstGenerationGenePool" -> tunerDef.tunerFirstGenerationGenePool,
      "tunerNumberOfGenerations" -> tunerDef.tunerNumberOfGenerations,
      "tunerNumberOfParentsToRetain" -> tunerDef.tunerNumberOfParentsToRetain,
      "tunerNumberOfMutationsPerGeneration" -> tunerDef.tunerNumberOfMutationsPerGeneration,
      "tunerGeneticMixing" -> tunerDef.tunerGeneticMixing,
      "tunerGenerationalMutationStrategy" -> tunerDef.tunerGenerationalMutationStrategy,
      "tunerFixedMutationValue" -> tunerDef.tunerFixedMutationValue,
      "tunerMutationMagnitudeMode" -> tunerDef.tunerMutationMagnitudeMode,
      "tunerEvolutionStrategy" -> tunerDef.tunerEvolutionStrategy,
      "tunerGeneticMBORegressorType" -> tunerDef.tunerGeneticMBORegressorType,
      "tunerGeneticMBOCandidateFactor" -> tunerDef.tunerGeneticMBOCandidateFactor,
      "tunerContinuousEvolutionImprovementThreshold" -> tunerDef.tunerContinuousEvolutionImprovementThreshold,
      "tunerContinuousEvolutionMaxIterations" -> tunerDef.tunerContinuousEvolutionMaxIterations,
      "tunerContinuousEvolutionStoppingScore" -> tunerDef.tunerContinuousEvolutionStoppingScore,
      "tunerContinuousEvolutionParallelism" -> tunerDef.tunerContinuousEvolutionParallelism,
      "tunerContinuousEvolutionMutationAggressiveness" -> tunerDef.tunerContinuousEvolutionMutationAggressiveness,
      "tunerContinuousEvolutionGeneticMixing" -> tunerDef.tunerContinuousEvolutionGeneticMixing,
      "tunerContinuousEvolutionRollingImprovementCount" -> tunerDef.tunerContinuousEvolutionRollingImprovingCount,
      "tunerModelSeed" -> tunerDef.tunerModelSeed,
      "tunerHyperSpaceInferenceFlag" -> tunerDef.tunerHyperSpaceInference,
      "tunerHyperSpaceInferenceCount" -> tunerDef.tunerHyperSpaceInferenceCount,
      "tunerHyperSpaceModelCount" -> tunerDef.tunerHyperSpaceModelCount,
      "tunerHyperSpaceModelType" -> tunerDef.tunerHyperSpaceModelType,
      "tunerInitialGenerationMode" -> tunerDef.tunerInitialGenerationMode,
      "tunerInitialGenerationPermutationCount" -> tunerDef.tunerInitialGenerationPermutationCount,
      "tunerInitialGenerationIndexMixingMode" -> tunerDef.tunerInitialGenerationIndexMixingMode,
      "tunerInitialGenerationArraySeed" -> tunerDef.tunerInitialGenerationArraySeed,
      "tunerOutputDfRepartitionScaleFactor" -> tunerDef.tunerOutputDfRepartitionScaleFactor,
      "mlFlowLoggingFlag" -> logDef.mlFlowLoggingFlag,
      "mlFlowLogArtifactsFlag" -> logDef.mlFlowLogArtifactsFlag,
      "mlFlowTrackingURI" -> logDef.mlFlowTrackingURI,
      "mlFlowExperimentName" -> logDef.mlFlowExperimentName,
      "mlFlowAPIToken" -> logDef.mlFlowAPIToken,
      "mlFlowModelSaveDirectory" -> logDef.mlFlowModelSaveDirectory,
      "mlFlowLoggingMode" -> logDef.mlFlowLoggingMode,
      "mlFlowBestSuffix" -> logDef.mlFlowBestSuffix,
      "inferenceConfigSaveLocation" -> logDef.inferenceConfigSaveLocation,
      "mlFlowCustomRunTags" -> logDef.mlFlowCustomRunTags
    )
  }

}
