package com.databricks.labs.automl.executor.config

class InstanceConfigValidation(modelFamily: String,
                               modelType: String,
                               configs: Array[InstanceConfig])
    extends ConfigurationDefaults {

  final val OHE_WARNING = "One hot encoding " +
    "is not recommended for tree-based algorithms.  In order to learn features, depending on cardinality of " +
    "the categorical one hot encoding column, the depth of the trees would need to be very large.  This risks" +
    "building a poorly fit model.  Proceeed only if you understand the implications."
  final val SCALING_WARNING
    : String = "For a non-tree based model, failing to scale the feature vector could lead" +
    "to a model that fits erroneously to values that are of larger scaled magnitude that is not intended to influence" +
    "the model.  "

  //Scoring Metric test

  //

  final val MODEL_FAMILY = familyTypeEvaluator(modelFamily)

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

  private def validateSwitchConfigs(modelFamily: String,
                                    switchConfig: SwitchConfig): Unit = {

    import FamilyValidator._

    // Tree validation checks
    MODEL_FAMILY match {
      case Trees =>
        if (switchConfig.oneHotEncodeFlag)
          warningPrint("oneHotEncodeFlag", OHE_WARNING, "warn")
      case NonTrees =>
        if (!switchConfig.scalingFlag)
          warningPrint("scalingFlag", SCALING_WARNING, "warn")
    }

  }

  private def validateTunerConfigs(modelFamily: String,
                                   tunerConfig: TunerConfig): Boolean = ???

}

object InstanceConfigValidation {

  def apply(modelFamily: String,
            modelType: String,
            configs: Array[InstanceConfig]): Unit =
    new InstanceConfigValidation(modelFamily, modelType, configs)

}
