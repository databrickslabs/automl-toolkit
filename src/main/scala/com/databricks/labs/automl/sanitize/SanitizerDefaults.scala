package com.databricks.labs.automl.sanitize

trait SanitizerDefaults {

  //TODO: fill in the rest of the default values here from the other packages within sanitize.

  /**
    * Global Defaults
    */
  def defaultLabelCol = "label"
  def defaultFeaturesCol = "features"

  /**
    * Pearson Defaults
    */
  final val _allowedStats: Array[String] =
    Array("pvalue", "degreesFreedom", "pearsonStat")
  final val _allowedFilterDirections: Array[String] = Array("greater", "lesser")
  final val _allowedFilterModes: Array[String] = Array("auto", "manual")

  def defaultPearsonFilterStatistic = "pvalue"
  def defaultPearsonFilterDirection = "greater"
  def defaultPearsonFilterManualValue = 0.0
  def defaultPearsonFilterMode = "auto"
  def defaultPearsonAutoFilterNTile = 0.99

  /**
    * Scaler Defaults
    */
  final val allowableScalers: Array[String] =
    Array("minMax", "standard", "normalize", "maxAbs")

  def defaultRenamedFeaturesCol = "features_r"
  def defaultScalerType = "minMax"
  def defaultScalerMin = 0.0
  def defaultScalerMax = 1.0
  def defaultStandardScalerMeanFlag = false
  def defaultStandardScalerStdDevFlag = true
  def defaultPNorm = 2.0

}
