package com.databricks.labs.automl.executor.config

import com.databricks.labs.automl.exceptions.MlFlowValidationException
import com.databricks.labs.automl.utils.WorkspaceDirectoryValidation

class InstanceConfigValidation(config: InstanceConfig)
    extends ConfigurationDefaults {

  final val OHE_WARNING: String = "One hot encoding " +
    "is not recommended for tree-based algorithms.  In order to learn features, depending on cardinality of " +
    "the categorical one hot encoding column, the depth of the trees would need to be very large.  This risks " +
    "building a poorly fit model.  Proceed only if you understand the implications."
  final val SCALING_WARNING
    : String = "For a non-tree based model, failing to scale the feature vector could lead" +
    "to a model that fits erroneously to values that are of larger scaled magnitude that is not intended to influence" +
    "the model.  "
  final val TRAIN_SPLIT_MODES: Array[String] = Array(
    "random",
    "chronological",
    "stratified",
    "overSample",
    "underSample",
    "stratifyReduce",
    "kSample"
  )
  final val INITIAL_GENERATION_MODES: Array[String] =
    Array("random", "permutations")
  final val INITIAL_GENERATION_MIXING_MODES: Array[String] =
    Array("random", "linear")
  final val GENETIC_MBO_REGRESSOR_TYPES: Array[String] =
    Array("XGBoost", "LinearRegression", "RandomForest")
  final val EVOLUTION_STRATEGIES: Array[String] = Array("batch", "continuous")
  final val OPTIMIZATION_STRATEGIES: Array[String] =
    Array("minimize", "maximize")
  final val MUTATION_STRATEGIES: Array[String] = Array("linear", "fixed")
  final val MUTATION_MAGNITUDE_MODES: Array[String] = Array("random", "fixed")
  final val REGRESSION_METRICS: Array[String] =
    Array("rmse", "mse", "r2", "mae")
  final val CLASSIFICATION_METRICS: Array[String] = Array(
    "f1",
    "weightedPrecision",
    "weightedRecall",
    "accuracy",
    "areaUnderPR",
    "areaUnderROC"
  )

  //TODO: finish validation checks

  final val MODEL_FAMILY = familyTypeEvaluator(config.modelFamily)

  private def warningPrint(settingName: String,
                           message: String,
                           level: String): Unit = {

    val levelString = level match {
      case "warn" => "[WARNING] "
      case "info" => "[INFO] "
      case _      => ""
    }

    println(s"$levelString The configuration key '$settingName' $message")

  }

  private def assertionPrint(settingName: String,
                             setStringParameter: String): Unit = {

    val message =
      s"[ERROR] Value provided for $settingName : ($setStringParameter) is not a member of "

    settingName match {
      case "tunerTrainSplitMethod" =>
        assert(
          TRAIN_SPLIT_MODES.contains(setStringParameter),
          message + TRAIN_SPLIT_MODES.mkString(", ")
        )
      case "tunerInitialGenerationMode" =>
        assert(
          INITIAL_GENERATION_MODES.contains(setStringParameter),
          message + INITIAL_GENERATION_MODES.mkString(", ")
        )
      case "tunerInitialGenerationIndexMixingMode" =>
        assert(
          INITIAL_GENERATION_MIXING_MODES.contains(setStringParameter),
          message + INITIAL_GENERATION_MIXING_MODES.mkString(", ")
        )
      case "tunerGeneticMBORegressorType" =>
        assert(
          GENETIC_MBO_REGRESSOR_TYPES.contains(setStringParameter),
          message + GENETIC_MBO_REGRESSOR_TYPES.mkString(", ")
        )
      case "tunerEvolutionStrategy" =>
        assert(
          EVOLUTION_STRATEGIES.contains(setStringParameter),
          message + EVOLUTION_STRATEGIES.mkString(", ")
        )
      case "scoringOptimizationStrategy" =>
        assert(
          OPTIMIZATION_STRATEGIES.contains(setStringParameter),
          message + OPTIMIZATION_STRATEGIES.mkString(", ")
        )
      case "tunerGenerationalMutationStrategy" =>
        assert(
          MUTATION_STRATEGIES.contains(setStringParameter),
          message + MUTATION_STRATEGIES.mkString(", ")
        )
      case "tunerMutationMagnitudeMode" =>
        assert(
          MUTATION_MAGNITUDE_MODES.contains(setStringParameter),
          message + MUTATION_MAGNITUDE_MODES.mkString(", ")
        )
      case "scoringMetric" =>
        config.predictionType match {
          case "regressor" =>
            assert(
              REGRESSION_METRICS.contains(setStringParameter),
              message + REGRESSION_METRICS.mkString(", ")
            )
          case "classifier" =>
            assert(
              CLASSIFICATION_METRICS.contains(setStringParameter),
              message + CLASSIFICATION_METRICS.mkString(", ")
            )
        }
    }

  }

  /**
    * Warnings associated with configurations that might be problematic for some use cases relating to creation of the
    * feature vector.
    * @since 0.7.2
    * @author Ben Wilson, Databricks
    */
  private def validateSwitchConfigs(): Unit = {

    import FamilyValidator._

    // Tree validation checks
    MODEL_FAMILY match {
      case Trees =>
        if (config.switchConfig.oneHotEncodeFlag)
          warningPrint("oneHotEncodeFlag", OHE_WARNING, "warn")
      case NonTrees =>
        if (!config.switchConfig.scalingFlag)
          warningPrint("scalingFlag", SCALING_WARNING, "warn")
    }

  }

  /**
    * Early check on tuner configurations for potential conflicts that need to be validated prior to starting of the
    * Data Prep phase that are part of the modeling phases of the toolkit.
    * @since 0.7.2
    * @author Ben Wilson, Databricks
    */
  private def validateTunerConfigs(): Unit = {

    assertionPrint(
      "tunerTrainSplitMethod",
      config.tunerConfig.tunerTrainSplitMethod
    )
    assertionPrint(
      "tunerInitialGenerationMode",
      config.tunerConfig.tunerInitialGenerationMode
    )
    assertionPrint(
      "tunerInitialGenerationIndexMixingMode",
      config.tunerConfig.tunerInitialGenerationIndexMixingMode
    )
    assertionPrint(
      "tunerGeneticMBORegressorType",
      config.tunerConfig.tunerGeneticMBORegressorType
    )
    assertionPrint(
      "tunerEvolutionStrategy",
      config.tunerConfig.tunerEvolutionStrategy
    )
    assertionPrint(
      "scoringOptimizationStrategy",
      config.genericConfig.scoringOptimizationStrategy
    )
    assertionPrint(
      "tunerGenerationalMutationStrategy",
      config.tunerConfig.tunerGenerationalMutationStrategy
    )
    assertionPrint(
      "tunerMutationMagnitudeMode",
      config.tunerConfig.tunerMutationMagnitudeMode
    )
    assertionPrint("scoringMetric", config.genericConfig.scoringMetric)

  }

  /**
    * Method for checking the logging location for mlflow prior to the run start.  If the path does not exist in the
    * Workspace, then attempt to create it.
    * @since 0.7.2
    * @author Ben Wilson, Databricks
    */
  private def checkConfigurationPaths(): Unit = {

    val logConfig = config.loggingConfig

    val validatedWorkspace = WorkspaceDirectoryValidation(
      logConfig.mlFlowTrackingURI,
      logConfig.mlFlowAPIToken,
      logConfig.mlFlowExperimentName
    )

    val rootLogPath =
      logConfig.mlFlowExperimentName.split("/").dropRight(1).mkString("/")

    if (validatedWorkspace) {
      warningPrint(
        "mlFlowExperimentName",
        s"Logging directory validated at: $rootLogPath " +
          s"and will be logged to " +
          s"${logConfig.mlFlowExperimentName + "/" + config.modelFamily + "_" + config.predictionType}",
        "info"
      )

    } else {
      throw MlFlowValidationException(
        "Could not create Workspace directories.  Ensure your account has sufficient " +
          "permissions to create paths and write to this location by inspecting the Permissions tab within the " +
          s"Workspace Directory.  Path: $rootLogPath is not permitted to be accessed by your user account."
      )
    }

  }

  /**
    * Execute pre-run validation checks
    */
  def validate(): Unit = {

    validateSwitchConfigs()
    if(config.loggingConfig.mlFlowLoggingFlag) checkConfigurationPaths()
    validateTunerConfigs()

  }

}

object InstanceConfigValidation {

  def apply(config: InstanceConfig): InstanceConfigValidation =
    new InstanceConfigValidation(config)

}
