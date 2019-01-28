package com.databricks.spark.automatedml.inference

import com.databricks.spark.automatedml.params.{MLFlowConfig, ScalingConfig}

case class InferenceSwitchSettings(
                                    naFillFlag: Boolean,
                                    varianceFilterFlag: Boolean,
                                    outlierFilterFlag: Boolean,
                                    pearsonFilterFlag: Boolean,
                                    covarianceFilterFlag: Boolean,
                                    oneHotEncodeFlag: Boolean,
                                    scalingFlag: Boolean
                                  )

case class InferenceDataConfig(
                                labelCol: String,
                                featuresCol: String
                              )

case class InferenceModelConfig(
                                 modelFamily: String,
                                 modelType: String,
                                 modelLoadMethod: String,
                                 mlFlowConfig: MLFlowConfig,
                                 mlFlowRunId: String,
                                 modelPathLocation: String
                               )

case class NaFillConfig(
                         categoricalColumns: Map[String, String],
                         numericColumns: Map[String, Double]
                       )

case class VarianceFilterConfig(
                                 fieldsRemoved: Array[String]
                               )

case class OutlierFilteringConfig(
                                   fieldRemovalMap: Map[String, (Double, String)]
                                 )

case class CovarianceFilteringConfig(
                                      fieldsRemoved: Array[String]
                                    )

case class PearsonFilteringConfig(
                                   fieldsRemoved: Array[String]
                                 )

case class FeatureEngineeringConfig(
                                     naFillConfig: NaFillConfig,
                                     varianceFilterConfig: VarianceFilterConfig,
                                     outlierFilteringConfig: OutlierFilteringConfig,
                                     covarianceFilteringConfig: CovarianceFilteringConfig,
                                     pearsonFilteringConfig: PearsonFilteringConfig,
                                     scalingConfig: ScalingConfig
                                   )

case class InferenceMainConfig(
                                inferenceDataConfig: InferenceDataConfig,
                                inferenceSwitchSettings: InferenceSwitchSettings,
                                inferenceModelConfig: InferenceModelConfig,
                                featureEngineeringConfig: FeatureEngineeringConfig
                              )