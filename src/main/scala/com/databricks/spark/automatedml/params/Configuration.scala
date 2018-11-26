package com.databricks.spark.automatedml.params

case class MainConfig(
                       modelType: String,
                       labelCol: String ,
                       featuresCol: String,
                       naFillFlag: Boolean,
                       varianceFilterFlag: Boolean,
                       outlierFilterFlag: Boolean,
                       pearsonFilteringFlag: Boolean,
                       covarianceFilteringFlag: Boolean,
                       numericBoundaries: Map[String, (Double, Double)],
                       stringBoundaries: Map[String, List[String]],
                       scoringMetric: String,
                       scoringOptimizationStrategy: String,
                       fillConfig: FillConfig,
                       outlierConfig: OutlierConfig,
                       pearsonConfig: PearsonConfig,
                       covarianceConfig: CovarianceConfig,
                       geneticConfig: GeneticConfig
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

