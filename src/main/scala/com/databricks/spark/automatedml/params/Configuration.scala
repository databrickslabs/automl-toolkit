package com.databricks.spark.automatedml.params

case class MainConfig(
                       modelFamily: String,
                       labelCol: String,
                       featuresCol: String,
                       naFillFlag: Boolean,
                       varianceFilterFlag: Boolean,
                       outlierFilterFlag: Boolean,
                       pearsonFilteringFlag: Boolean,
                       covarianceFilteringFlag: Boolean,
                       scalingFlag: Boolean,
                       numericBoundaries: Map[String, (Double, Double)],
                       stringBoundaries: Map[String, List[String]],
                       scoringMetric: String,
                       scoringOptimizationStrategy: String,
                       fillConfig: FillConfig,
                       outlierConfig: OutlierConfig,
                       pearsonConfig: PearsonConfig,
                       covarianceConfig: CovarianceConfig,
                       scalingConfig: ScalingConfig,
                       geneticConfig: GeneticConfig
                     )

// TODO: Change MainConfig to use this case class definition.
case class DataPrepConfig(
                         naFillFlag: Boolean,
                         varianceFilterFlag: Boolean,
                         outlierFilterFlag: Boolean,
                         pearsonFilterFlag: Boolean,
                         covarianceFilterFlag: Boolean,
                         scalingFlag: Boolean
                         )

case class FillConfig(
                       numericFillStat: String,
                       characterFillStat: String,
                       modelSelectionDistinctThreshold: Int
                     )

case class OutlierConfig(
                          filterBounds: String,
                          lowerFilterNTile: Double,
                          upperFilterNTile: Double,
                          filterPrecision: Double,
                          continuousDataThreshold: Int,
                          fieldsToIgnore: Array[String]
                        )

case class PearsonConfig(
                          filterStatistic: String,
                          filterDirection: String,
                          filterManualValue: Double,
                          filterMode: String,
                          autoFilterNTile: Double
                        )

case class CovarianceConfig(
                             correlationCutoffLow: Double,
                             correlationCutoffHigh: Double
                           )

case class GeneticConfig(
                          parallelism: Int,
                          kFold: Int,
                          trainPortion: Double,
                          seed: Long,
                          firstGenerationGenePool: Int,
                          numberOfGenerations: Int,
                          numberOfParentsToRetain: Int,
                          numberOfMutationsPerGeneration: Int,
                          geneticMixing: Double,
                          generationalMutationStrategy: String,
                          fixedMutationValue: Int,
                          mutationMagnitudeMode: String
                        )

case class ScalingConfig(
                          scalerType: String,
                          scalerMin: Double,
                          scalerMax: Double,
                          standardScalerMeanFlag: Boolean,
                          standardScalerStdDevFlag: Boolean,
                          pNorm: Double
                        )