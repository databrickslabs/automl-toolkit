package com.databricks.labs.automl.inference

import com.databricks.labs.automl.feature.structures.InteractionPayloadExtract
import com.databricks.labs.automl.params.{MLFlowConfig, ScalingConfig}
import org.apache.spark.sql.DataFrame

case class InferenceSwitchSettings(naFillFlag: Boolean,
                                   varianceFilterFlag: Boolean,
                                   outlierFilterFlag: Boolean,
                                   pearsonFilterFlag: Boolean,
                                   covarianceFilterFlag: Boolean,
                                   oneHotEncodeFlag: Boolean,
                                   scalingFlag: Boolean,
                                   featureInteractionFlag: Boolean)

case class InferenceDataConfig(labelCol: String,
                               featuresCol: String,
                               startingColumns: Array[String],
                               fieldsToIgnore: Array[String],
                               dateTimeConversionType: String)

case class InferenceModelConfig(modelFamily: String,
                                modelType: String,
                                modelLoadMethod: String,
                                mlFlowConfig: MLFlowConfig,
                                mlFlowRunId: String,
                                modelPathLocation: String)

case class NaFillConfig(categoricalColumns: Map[String, String],
                        numericColumns: Map[String, Double],
                        booleanColumns: Map[String, Boolean])

case class NaFillPayload(categorical: Array[(String, Any)],
                         numeric: Array[(String, Any)],
                         boolean: Array[(String, Boolean)])

case class VarianceFilterConfig(fieldsRemoved: Array[String])

case class OutlierFilteringConfig(
  fieldRemovalMap: Map[String, (Double, String)]
)

case class CovarianceFilteringConfig(fieldsRemoved: Array[String])

case class PearsonFilteringConfig(fieldsRemoved: Array[String])

case class FeatureInteractionConfig(
  interactions: Array[InteractionPayloadExtract]
)

case class FeatureEngineeringConfig(
  naFillConfig: NaFillConfig,
  varianceFilterConfig: VarianceFilterConfig,
  outlierFilteringConfig: OutlierFilteringConfig,
  covarianceFilteringConfig: CovarianceFilteringConfig,
  pearsonFilteringConfig: PearsonFilteringConfig,
  scalingConfig: ScalingConfig,
  featureInteractionConfig: FeatureInteractionConfig
)

case class InferenceMainConfig(
  inferenceDataConfig: InferenceDataConfig,
  inferenceSwitchSettings: InferenceSwitchSettings,
  inferenceModelConfig: InferenceModelConfig,
  featureEngineeringConfig: FeatureEngineeringConfig,
  inferenceConfigStorageLocation: String
)

case class InferenceJsonReturn(compactJson: String, prettyJson: String)

trait InferenceBaseConstructor {
  def data: DataFrame
  def modelingColumns: Array[String]
  def allColumns: Array[String]
}

abstract case class InferencePayload() extends InferenceBaseConstructor
