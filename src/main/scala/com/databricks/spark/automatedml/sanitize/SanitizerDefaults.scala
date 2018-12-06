package com.databricks.spark.automatedml.sanitize

trait SanitizerDefaults {

  //TODO: fill in the rest of the default values here from the other packages within sanitize.

  /**
    * Global Defaults
    */

  def defaultFeaturesCol = "features"


  /**
    * Scaler Defaults
    */

  final val allowableScalers: Array[String] = Array("minMax", "standard", "normalize", "maxAbs")

  def defaultRenamedFeaturesCol = "features_r"
  def defaultScalerType = "minMax"
  def defaultScalerMin = 0.0
  def defaultScalerMax = 1.0
  def defaultStandardScalerMeanFlag = false
  def defaultStandardScalerStdDevFlag = true
  def defaultPNorm = 2.0


}
