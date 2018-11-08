package com.databricks.spark.automatedml

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.regression.LinearRegressionModel


case class PearsonPayload(fieldName: String,
                          pvalue: Double,
                          degreesFreedom: Int,
                          pearsonStat: Double)

case class FilterData(
                       field: String,
                       uniqueValues: Long
                     )

case class ManualFilters(
                          field: String,
                          threshold: Double
                        )

case class RandomForestConfig(numTrees: Int,
                              impurity: String,
                              maxBins: Int,
                              maxDepth: Int,
                              minInfoGain: Double,
                              subSamplingRate: Double,
                              featureSubsetStrategy: String
                             )

case class LogisticRegressionConfig(
                                     elasticNetParams: Double,
                                     fitIntercept: Boolean,
                                     maxIter: Int,
                                     regParam: Double,
                                     standardization: Boolean,
                                     tolerance: Double
                                   )

case class LinearRegressionConfig(
                                   elasticNetParams: Double,
                                   fitIntercept: Boolean,
                                   loss: String,
                                   maxIter: Int,
                                   regParam: Double,
                                   standardization: Boolean,
                                   tolerance: Double
                                 )

case class LinearRegressionModelsWithResults(
                                              modelHyperParams: LinearRegressionConfig,
                                              model: LinearRegressionModel,
                                              score: Double,
                                              evalMetrics: Map[String, Double],
                                              generation: Int
                                            )

case class LogisticRegressionModelsWithResults(
                                                modelHyperParams: LogisticRegressionConfig,
                                                model: LogisticRegressionModel,
                                                score: Double,
                                                evalMetrics: Map[String, Double],
                                                generation: Int
                                              )

case class RandomForestModelsWithResults(
                                          modelHyperParams: RandomForestConfig,
                                          model: Any,
                                          score: Double,
                                          evalMetrics: Map[String, Double],
                                          generation: Int
                                        )

case class LogisticModelsWithResults(
                                      modelHyperParams: LogisticRegressionConfig,
                                      model: LogisticRegressionModel,
                                      score: Double,
                                      evalMetrics: Map[String, Double],
                                      generation: Int
                                    )

case class LinearModelsWithResults(
                                    modelHyperParams: LinearRegressionConfig,
                                    model: LinearRegressionModel,
                                    score: Double,
                                    evalMetrics: Map[String, Double],
                                    generation: Int
                                  )

case class StaticModelConfig(
                              labelColumn: String,
                              featuresColumn: String
                            )

sealed trait ModelType[A, B]

final case class ClassiferType[A, B](a: A) extends ModelType[A, B]

final case class RegressorType[A, B](b: B) extends ModelType[A, B]


