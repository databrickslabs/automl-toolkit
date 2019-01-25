package com.databricks.spark.automatedml.inference

trait InferenceConfig extends InferenceDefaults{

  final val allowableModelLoads: Array[String] = Array("path", "mlflow")
  final val allowableOutlierFilteringDirections: Array[String] = Array("greater", "lesser")

  var _inferenceDataConfig: InferenceDataConfig = _defaultInferenceDataConfig
  var _inferenceSwitchSettings: InferenceSwitchSettings = _defaultInferenceSwitchSettings
  var _inferenceModelConfig: InferenceModelConfig = _defaultInferenceModelConfig
  var _featureEngineeringConfig: FeatureEngineeringConfig = _defaultFeatureEngineeringConfig

  var _inferenceConfig: InferenceMainConfig = _defaultInferenceConfig


 //TODO: setters for information that needs to be passed in for recording configuration?
  // The model data in particular needs setters.  Everything else seems ok for case class setters.

    def setInferenceConfig(value: InferenceMainConfig): this.type = {
      _inferenceConfig = value
      this
    }

  def setInferenceConfig(): this.type = {
    _inferenceConfig = InferenceMainConfig(
      inferenceDataConfig = _inferenceDataConfig,
      inferenceSwitchSettings = _inferenceSwitchSettings,
      inferenceModelConfig = _inferenceModelConfig,
      featureEngineeringConfig = _featureEngineeringConfig
    )
    this
  }

  def setInferenceSwitchSettings(value: InferenceSwitchSettings): this.type = {
    _inferenceSwitchSettings = value

  }

  def getInferenceConfig: InferenceMainConfig = _inferenceConfig
}
