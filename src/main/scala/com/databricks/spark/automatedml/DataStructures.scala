package com.databricks.spark.automatedml

import org.apache.spark.ml.classification.{LinearSVCModel, LogisticRegressionModel}
import org.apache.spark.ml.regression.LinearRegressionModel


//TODO: add in ability to do Bayesian hyperparameter search for large data sets
//TODO: add in time logging based on each model's runtime to each case class for ModelsWithResults
//TODO: main entry points, main runner object
//TODO: split out each core functionality package to its own directory


case class PearsonPayload(
                           fieldName: String,
                           pvalue: Double,
                           degreesFreedom: Int,
                           pearsonStat: Double
                         )

case class FeatureCorrelationStats(
                                    leftCol: String,
                                    rightCol: String,
                                    correlation: Double
                                  )

case class FilterData(
                       field: String,
                       uniqueValues: Long
                     )

case class ManualFilters(
                          field: String,
                          threshold: Double
                        )

case class RandomForestConfig(
                               numTrees: Int,
                               impurity: String,
                               maxBins: Int,
                               maxDepth: Int,
                               minInfoGain: Double,
                               subSamplingRate: Double,
                               featureSubsetStrategy: String
                             )

case class GBTConfig(
                      impurity: String,
                      lossType: String,
                      maxBins: Int,
                      maxDepth: Int,
                      maxIter: Int,
                      minInfoGain: Double,
                      minInstancesPerNode: Int,
                      stepSize: Double
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

case class GBTModelsWithResults(
                                 modelHyperParams: GBTConfig,
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


case class SVMConfig(
                      fitIntercept: Boolean,
                      maxIter: Int,
                      regParam: Double,
                      standardization: Boolean,
                      tol: Double
                    )

case class SVMModelsWithResults(
                                 modelHyperParams: SVMConfig,
                                 model: LinearSVCModel,
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


