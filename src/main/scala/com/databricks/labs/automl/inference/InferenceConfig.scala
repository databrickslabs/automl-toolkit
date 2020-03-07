package com.databricks.labs.automl.inference

import com.databricks.labs.automl.params.{MLFlowConfig, ScalingConfig}

object InferenceConfig extends InferenceDefaults {

  final val allowableModelLoads: Array[String] = Array("path", "mlflow")
  final val allowableOutlierFilteringDirections: Array[String] =
    Array("greater", "lesser")

  var _inferenceConfigStorageLocation: String = ""
  var _inferenceDataConfig: InferenceDataConfig = _defaultInferenceDataConfig
  var _inferenceDataConfigLabelCol: String =
    _defaultInferenceDataConfig.labelCol
  var _inferenceDataConfigFeaturesCol: String =
    _defaultInferenceDataConfig.featuresCol
  var _inferenceDataConfigStartingColumns: Array[String] =
    _defaultInferenceDataConfig.startingColumns
  var _inferenceDataConfigFieldsToIgnore: Array[String] =
    _defaultInferenceDataConfig.fieldsToIgnore
  var _inferenceDataConfigDateTimeConversionType: String =
    _defaultInferenceDataConfig.dateTimeConversionType
  var _inferenceSwitchSettings: InferenceSwitchSettings =
    _defaultInferenceSwitchSettings
  var _inferenceModelConfig: InferenceModelConfig = _defaultInferenceModelConfig
  var _featureEngineeringConfig: FeatureEngineeringConfig =
    _defaultFeatureEngineeringConfig
  var _inferenceConfig: InferenceMainConfig = _defaultInferenceConfig
  var _inferenceConfigModelFamily: String =
    _defaultInferenceModelConfig.modelFamily
  var _inferenceConfigModelType: String = _defaultInferenceModelConfig.modelType
  var _inferenceConfigModelLoadMethod: String =
    _defaultInferenceModelConfig.modelLoadMethod
  var _inferenceConfigMlFlowConfig: MLFlowConfig =
    _defaultInferenceModelConfig.mlFlowConfig
  var _inferenceConfigMlFlowRunId: String =
    _defaultInferenceModelConfig.mlFlowRunId
  var _inferenceConfigModelPathLocation: String =
    _defaultInferenceModelConfig.modelPathLocation
  var _inferenceConfigMlFlowTrackingURI: String =
    _defaultInferenceModelConfig.mlFlowConfig.mlFlowTrackingURI
  var _inferenceConfigMlFlowExperimentName: String =
    _defaultInferenceModelConfig.mlFlowConfig.mlFlowExperimentName
  var _inferenceConfigMlFlowAPIToken: String =
    _defaultInferenceModelConfig.mlFlowConfig.mlFlowAPIToken
  var _inferenceConfigMlFlowModelSaveDirectory: String =
    _defaultInferenceModelConfig.mlFlowConfig.mlFlowModelSaveDirectory
  var _inferenceNaFillConfig: NaFillConfig = _defaultNaFillConfig
  var _inferenceVarianceFilterConfig: VarianceFilterConfig =
    _defaultVarianceFilterConfig
  var _inferenceOutlierFilteringConfig: OutlierFilteringConfig =
    _defaultOutlierFilteringConfig
  var _inferenceCovarianceFilteringConfig: CovarianceFilteringConfig =
    _defaultCovarianceFilteringConfig
  var _inferencePearsonFilteringConfig: PearsonFilteringConfig =
    _defaultPearsonFilteringConfig
  var _inferenceScalingConfig: ScalingConfig = _scalingConfigDefaults
  var _inferenceScalerType: String = _scalingConfigDefaults.scalerType
  var _inferenceScalerMin: Double = _scalingConfigDefaults.scalerMin
  var _inferenceScalerMax: Double = _scalingConfigDefaults.scalerMax
  var _inferenceStandardScalerMeanFlag: Boolean =
    _scalingConfigDefaults.standardScalerMeanFlag
  var _inferenceStandardScalerStdDevFlag: Boolean =
    _scalingConfigDefaults.standardScalerStdDevFlag
  var _inferenceScalerPNorm: Double = _scalingConfigDefaults.pNorm
  var _inferenceFeatureInteractionConfig: FeatureInteractionConfig =
    _defaultInferenceFeatureInteractionConfig

  def setInferenceConfigStorageLocation(value: String): this.type = {
    _inferenceConfigStorageLocation = value
    setInferenceConfig()
    this
  }

  def setInferenceConfig(value: InferenceMainConfig): this.type = {
    _inferenceConfig = value
    this
  }

  def setInferenceConfig(): this.type = {
    _inferenceConfig = InferenceMainConfig(
      inferenceDataConfig = _inferenceDataConfig,
      inferenceSwitchSettings = _inferenceSwitchSettings,
      inferenceModelConfig = _inferenceModelConfig,
      featureEngineeringConfig = _featureEngineeringConfig,
      inferenceConfigStorageLocation = _inferenceConfigStorageLocation
    )
    this
  }

  def setInferenceSwitchSettings(value: InferenceSwitchSettings): this.type = {
    _inferenceSwitchSettings = value
    setInferenceConfig()
    this
  }

  def setInferenceDataConfig(value: InferenceDataConfig): this.type = {
    _inferenceDataConfig = value
    _inferenceDataConfigLabelCol = value.labelCol
    _inferenceDataConfigFeaturesCol = value.featuresCol
    _inferenceDataConfigStartingColumns = value.startingColumns
    _inferenceDataConfigFieldsToIgnore = value.fieldsToIgnore
    _inferenceDataConfigDateTimeConversionType = value.dateTimeConversionType
    setInferenceConfig()
    this
  }

  private def setInferenceDataConfig(): this.type = {
    _inferenceDataConfig = InferenceDataConfig(
      labelCol = _inferenceDataConfigLabelCol,
      featuresCol = _inferenceDataConfigFeaturesCol,
      startingColumns = _inferenceDataConfigStartingColumns,
      fieldsToIgnore = _inferenceDataConfigFieldsToIgnore,
      dateTimeConversionType = _inferenceDataConfigDateTimeConversionType
    )
    setInferenceConfig()
    this
  }

  def setInferenceDataConfigLabelCol(value: String): this.type = {
    _inferenceDataConfigLabelCol = value
    setInferenceDataConfig()
    this
  }

  def setInferenceDataConfigFeaturesCol(value: String): this.type = {
    _inferenceDataConfigFeaturesCol = value
    setInferenceDataConfig()
    this
  }

  def setInferenceDataConfigStartingColumns(value: Array[String]): this.type = {
    _inferenceDataConfigStartingColumns = value
    setInferenceDataConfig()
    this
  }

  def setInferenceDataConfigFieldsToIgnore(value: Array[String]): this.type = {
    _inferenceDataConfigFieldsToIgnore = value
    setInferenceDataConfig()
    this
  }

  def setInferenceDataConfigDateTimeConversionType(value: String): this.type = {
    _inferenceDataConfigDateTimeConversionType = value
    setInferenceDataConfig()
    this
  }

  def setInferenceModelConfig(value: InferenceModelConfig): this.type = {
    _inferenceModelConfig = value
    setInferenceConfig()
    this
  }

  private def setInferenceModelConfig(): this.type = {
    _inferenceModelConfig = InferenceModelConfig(
      modelFamily = _inferenceConfigModelFamily,
      modelType = _inferenceConfigModelType,
      modelLoadMethod = _inferenceConfigModelLoadMethod,
      mlFlowConfig = _inferenceConfigMlFlowConfig,
      mlFlowRunId = _inferenceConfigMlFlowRunId,
      modelPathLocation = _inferenceConfigModelPathLocation
    )
    setInferenceConfig()
    this
  }

  def setInferenceModelConfigModelFamily(value: String): this.type = {
    _inferenceConfigModelFamily = value
    setInferenceModelConfig()
    this
  }

  def setInferenceModelConfigModelType(value: String): this.type = {
    _inferenceConfigModelType = value
    setInferenceModelConfig()
    this
  }

  def setInferenceModelConfigModelLoadMethod(value: String): this.type = {
    require(
      allowableModelLoads.contains(value),
      s"Inference Model Config Model Load Method invalid '$value' is not " +
        s"in ${allowableModelLoads.mkString(", ")}"
    )
    _inferenceConfigModelLoadMethod = value
    setInferenceModelConfig()
    this
  }

  def setInferenceModelConfigMlFlowConfig(value: MLFlowConfig): this.type = {
    _inferenceConfigMlFlowConfig = value
    setInferenceModelConfig()
    this
  }

  private def setInferenceModelConfigMlFlowConfig(): this.type = {
    _inferenceConfigMlFlowConfig = MLFlowConfig(
      mlFlowTrackingURI = _inferenceConfigMlFlowTrackingURI,
      mlFlowExperimentName = _inferenceConfigMlFlowExperimentName,
      mlFlowAPIToken = _inferenceConfigMlFlowAPIToken,
      mlFlowModelSaveDirectory = _inferenceConfigMlFlowModelSaveDirectory,
      mlFlowLoggingMode = "full",
      mlFlowBestSuffix = "_best",
      mlFlowCustomRunTags = Map("" -> "")
    )
    setInferenceModelConfig()
    this
  }

  def setInferenceConfigMlFlowTrackingURI(value: String): this.type = {
    _inferenceConfigMlFlowTrackingURI = value
    setInferenceModelConfigMlFlowConfig()
    this
  }

  def setInferenceConfigMlFlowExperimentName(value: String): this.type = {
    _inferenceConfigMlFlowExperimentName = value
    setInferenceModelConfigMlFlowConfig()
    this
  }

  def setInferenceConfigMlFlowAPIToken(value: String): this.type = {
    _inferenceConfigMlFlowAPIToken = value
    setInferenceModelConfigMlFlowConfig()
    this
  }

  def setInferenceConfigMlFlowModelSaveDirectory(value: String): this.type = {
    _inferenceConfigMlFlowModelSaveDirectory = value
    setInferenceModelConfigMlFlowConfig()
    this
  }

  def setInferenceModelConfigMlFlowRunID(value: String): this.type = {
    _inferenceConfigMlFlowRunId = value
    setInferenceModelConfig()
    this
  }

  def setInferenceModelConfigModelPathLocation(value: String): this.type = {
    _inferenceConfigModelPathLocation = value
    setInferenceModelConfig()
    this
  }

  def setInferenceFeatureEngineeringConfig(
    value: FeatureEngineeringConfig
  ): this.type = {
    _featureEngineeringConfig = value
    setInferenceConfig()
    this
  }

  private def setInferenceFeatureEngineeringConfig(): this.type = {
    _featureEngineeringConfig = FeatureEngineeringConfig(
      naFillConfig = _inferenceNaFillConfig,
      varianceFilterConfig = _inferenceVarianceFilterConfig,
      outlierFilteringConfig = _inferenceOutlierFilteringConfig,
      covarianceFilteringConfig = _inferenceCovarianceFilteringConfig,
      pearsonFilteringConfig = _inferencePearsonFilteringConfig,
      scalingConfig = _inferenceScalingConfig,
      featureInteractionConfig = _inferenceFeatureInteractionConfig
    )
    setInferenceConfig()
    this
  }

  def setInferenceNaFillConfig(categoricalMap: Map[String, String],
                               numericMap: Map[String, Double],
                               booleanMap: Map[String, Boolean]): this.type = {
    _inferenceNaFillConfig = NaFillConfig(
      categoricalColumns = categoricalMap,
      numericColumns = numericMap,
      booleanColumns = booleanMap
    )
    setInferenceFeatureEngineeringConfig()
    this
  }

  def setInferenceVarianceFilterConfig(value: Array[String]): this.type = {
    _inferenceVarianceFilterConfig = VarianceFilterConfig(fieldsRemoved = value)
    setInferenceFeatureEngineeringConfig()
    this
  }

  def setInferenceOutlierFilteringConfig(
    value: Map[String, (Double, String)]
  ): this.type = {
    _inferenceOutlierFilteringConfig = OutlierFilteringConfig(
      fieldRemovalMap = value
    )
    setInferenceFeatureEngineeringConfig()
    this
  }

  def setInferenceCovarianceFilteringConfig(value: Array[String]): this.type = {
    _inferenceCovarianceFilteringConfig = CovarianceFilteringConfig(
      fieldsRemoved = value
    )
    setInferenceFeatureEngineeringConfig()
    this
  }

  def setInferencePearsonFilteringConfig(value: Array[String]): this.type = {
    _inferencePearsonFilteringConfig = PearsonFilteringConfig(
      fieldsRemoved = value
    )
    setInferenceFeatureEngineeringConfig()
    this
  }

  private def setInferenceScalingConfig(): this.type = {
    _inferenceScalingConfig = ScalingConfig(
      scalerType = _inferenceScalerType,
      scalerMin = _inferenceScalerMin,
      scalerMax = _inferenceScalerMax,
      standardScalerMeanFlag = _inferenceStandardScalerMeanFlag,
      standardScalerStdDevFlag = _inferenceStandardScalerStdDevFlag,
      pNorm = _inferenceScalerPNorm
    )
    setInferenceConfig()
    this
  }

  def setFeatureInteractionConfig(
    value: FeatureInteractionConfig
  ): this.type = {
    _inferenceFeatureInteractionConfig = value
    setInferenceFeatureEngineeringConfig()
    this
  }

  def setInferenceScalingConfig(value: ScalingConfig): this.type = {
    _inferenceScalingConfig = value
    setInferenceConfig()
    this
  }

  def setInferenceScalerType(value: String): this.type = {
    _inferenceScalerType = value
    setInferenceScalingConfig()
    this
  }

  def setInferenceScalerMin(value: Double): this.type = {
    _inferenceScalerMin = value
    setInferenceScalingConfig()
    this
  }

  def setInferenceScalerMax(value: Double): this.type = {
    _inferenceScalerMax = value
    setInferenceScalingConfig()
    this
  }

  def setInferenceStandardScalerMeanFlagOn(): this.type = {
    _inferenceStandardScalerMeanFlag = true
    setInferenceScalingConfig()
    this
  }

  def setInferenceStandardScalerMeanFlagOff(): this.type = {
    _inferenceStandardScalerMeanFlag = false
    setInferenceScalingConfig()
    this
  }

  def setInferenceStandardScalerStdDevFlagOn(): this.type = {
    _inferenceStandardScalerStdDevFlag = true
    setInferenceScalingConfig()
    this
  }

  def setInferenceStandardScalerStdDevFlagOff(): this.type = {
    _inferenceStandardScalerStdDevFlag = false
    setInferenceScalingConfig()
    this
  }

  def setInferenceScalerPNorm(value: Double): this.type = {
    _inferenceScalerPNorm = value
    setInferenceScalingConfig()
    this
  }

  def getInferenceConfigStorageLocation: String =
    _inferenceConfigStorageLocation

  def getInferenceConfig: InferenceMainConfig = _inferenceConfig

  def getInferenceSwitchSettings: InferenceSwitchSettings =
    _inferenceSwitchSettings

  def getInferenceDataConfig: InferenceDataConfig = _inferenceDataConfig

  def getInferenceDataConfigLabelCol: String = _inferenceDataConfigLabelCol

  def getInferenceDataConfigFeaturesCol: String =
    _inferenceDataConfigFeaturesCol

  def getInferenceDataConfigStartingColumns: Array[String] =
    _inferenceDataConfigStartingColumns

  def getInferenceDataConfigFieldsToIgnore: Array[String] =
    _inferenceDataConfigFieldsToIgnore

  def getInferenceDataConfigDateTimeConversionType: String =
    _inferenceDataConfigDateTimeConversionType

  def getInferenceModelConfig: InferenceModelConfig = _inferenceModelConfig

  def getInferenceModelConfigModelFamily: String = _inferenceConfigModelFamily

  def getInferenceModelConfigModelType: String = _inferenceConfigModelType

  def getInferenceModelConfigModelLoadMethod: String =
    _inferenceConfigModelLoadMethod

  def getInferenceModelConfigMlFlowTrackingURI: String =
    _inferenceConfigMlFlowTrackingURI

  def getInferenceModelConfigMlFlowExperimentName: String =
    _inferenceConfigMlFlowExperimentName

  def getInferenceModelConfigMlFlowModelSaveDirectory: String =
    _inferenceConfigMlFlowModelSaveDirectory

  def getInferenceModelConfigMlFlowRunID: String = _inferenceConfigMlFlowRunId

  def getInferenceModelConfigModelPathLocation: String =
    _inferenceConfigModelPathLocation

  def getInferenceFeatureEngineeringConfig: FeatureEngineeringConfig =
    _featureEngineeringConfig

  def getInferenceNaFillConfig: NaFillConfig = _inferenceNaFillConfig

  def getInferenceVarianceFilterConfig: VarianceFilterConfig =
    _inferenceVarianceFilterConfig

  def getInferenceOutlierFilteringConfig: OutlierFilteringConfig =
    _inferenceOutlierFilteringConfig

  def getInferenceCovarianceFilteringConfig: CovarianceFilteringConfig =
    _inferenceCovarianceFilteringConfig

  def getInferencePearsonFilteringConfig: PearsonFilteringConfig =
    _inferencePearsonFilteringConfig

  def getInferenceScalingConfig: ScalingConfig = _inferenceScalingConfig

  def getInferenceScalerType: String = _inferenceScalerType

  def getInferenceScalerMin: Double = _inferenceScalerMin

  def getInferenceScalerMax: Double = _inferenceScalerMax

  def getInferenceStandardScalerMeanFlag: Boolean =
    _inferenceStandardScalerMeanFlag

  def getInferenceStandardScalerStdDevFlag: Boolean =
    _inferenceStandardScalerStdDevFlag

  def getInferenceScalerPNorm: Double = _inferenceScalerPNorm

  def getInferenceFeatureInteractionConfig: FeatureInteractionConfig =
    _inferenceFeatureInteractionConfig

}
//object InferenceConfig extends InferenceConfig{
//}
