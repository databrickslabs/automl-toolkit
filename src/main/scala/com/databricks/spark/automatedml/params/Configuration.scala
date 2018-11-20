package com.databricks.spark.automatedml.params

import org.apache.spark.sql.DataFrame


//public
case class MainConfig(
                       modelType: String = "RandomForest",
                       df: DataFrame,
                       labelCol: String = "label",
                       featuresCol: String = "features",
                       naFillFlag: Boolean = true,
                       varianceFilterFlag: Boolean = true,
                       outlierFilterFlag: Boolean = true,
                       pearsonFilteringFlag: Boolean = true,
                       covarianceFilteringFlag: Boolean = true,
                       numericBoundaries: Option[Map[String, (Double, Double)]] = None,
                       stringBoundaries: Option[Map[String, List[String]]] = None,
                       scoringMetric: Option[String] = None,
                       scoringOptimizationStrategy: Option[String] = None,
                       fillConfig: Option[FillConfig] = None,
                       outlierConfig: Option[OutlierConfig] = None,
                       pearsonConfig: Option[PearsonConfig] = None,
                       covarianceConfig: Option[CovarianceConfig] = None,
                       geneticConfig: Option[GeneticConfig] = None
                     )

// public
case class FillConfig(
                       numericFillStat: String,
                       characterFillStat: String,
                       modelSelectionDistinctThreshold: Int
                     )

// public
case class OutlierConfig(
                          filterBounds: String,
                          lowerFilterNTile: Double,
                          upperFilterNTile: Double,
                          filterPrecision: Double,
                          continuousDataThreshold: Int,
                          fieldsToIgnore: Array[String]
                        )

//public
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

// public
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

