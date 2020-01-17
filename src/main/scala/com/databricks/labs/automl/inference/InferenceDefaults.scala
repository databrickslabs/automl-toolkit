package com.databricks.labs.automl.inference

import com.databricks.labs.automl.feature.structures.InteractionPayloadExtract
import com.databricks.labs.automl.params.Defaults

trait InferenceDefaults extends Defaults {

  def _defaultInferenceSwitchSettings: InferenceSwitchSettings =
    InferenceSwitchSettings(
      naFillFlag = true,
      varianceFilterFlag = true,
      outlierFilterFlag = false,
      pearsonFilterFlag = false,
      covarianceFilterFlag = false,
      oneHotEncodeFlag = false,
      scalingFlag = false,
      featureInteractionFlag = false
    )

  def _defaultInferenceDataConfig: InferenceDataConfig = InferenceDataConfig(
    labelCol = "label",
    featuresCol = "features",
    startingColumns = Array.empty[String],
    fieldsToIgnore = Array.empty[String],
    dateTimeConversionType = "split"
  )

  def _defaultInferenceModelConfig: InferenceModelConfig = InferenceModelConfig(
    modelFamily = "RandomForest",
    modelType = "classifier",
    modelLoadMethod = "path",
    mlFlowConfig = _mlFlowConfigDefaults,
    mlFlowRunId = "a",
    modelPathLocation = "/models/"
  )

  def _defaultNaFillConfig: NaFillConfig = NaFillConfig(
    categoricalColumns = Map("default" -> "default"),
    numericColumns = Map("default_num" -> 0.0),
    booleanColumns = Map("default_bool" -> false)
  )

  def _defaultVarianceFilterConfig: VarianceFilterConfig = VarianceFilterConfig(
    fieldsRemoved = Array.empty[String]
  )

  def _defaultOutlierFilteringConfig: OutlierFilteringConfig =
    OutlierFilteringConfig(
      fieldRemovalMap = Map("" -> (Double.MaxValue, "greater"))
    )

  def _defaultCovarianceFilteringConfig: CovarianceFilteringConfig =
    CovarianceFilteringConfig(fieldsRemoved = Array.empty[String])

  def _defaultPearsonFilteringConfig: PearsonFilteringConfig =
    PearsonFilteringConfig(fieldsRemoved = Array.empty[String])

  def _defaultInferenceFeatureInteractionConfig: FeatureInteractionConfig =
    FeatureInteractionConfig(interactions = Array[InteractionPayloadExtract]())

  def _defaultFeatureEngineeringConfig: FeatureEngineeringConfig =
    FeatureEngineeringConfig(
      naFillConfig = _defaultNaFillConfig,
      varianceFilterConfig = _defaultVarianceFilterConfig,
      outlierFilteringConfig = _defaultOutlierFilteringConfig,
      covarianceFilteringConfig = _defaultCovarianceFilteringConfig,
      pearsonFilteringConfig = _defaultPearsonFilteringConfig,
      scalingConfig = _scalingConfigDefaults,
      featureInteractionConfig = _defaultInferenceFeatureInteractionConfig
    )

  def _defaultInferenceConfig: InferenceMainConfig = InferenceMainConfig(
    inferenceDataConfig = _defaultInferenceDataConfig,
    inferenceSwitchSettings = _defaultInferenceSwitchSettings,
    inferenceModelConfig = _defaultInferenceModelConfig,
    featureEngineeringConfig = _defaultFeatureEngineeringConfig,
    inferenceConfigStorageLocation = ""
  )

}
