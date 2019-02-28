package com.databricks.spark.automatedml.params

import org.apache.spark.ml.classification.{LinearSVCModel, LogisticRegressionModel, MultilayerPerceptronClassificationModel}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.DataFrame

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

case class XGBoostConfig(
                        alpha: Double,
                        eta: Double,
                        gamma: Double,
                        lambda: Double,
                        maxDepth: Int,
                        subSample: Double,
                        minChildWeight: Double,
                        numRound: Int,
                        maxBins: Int,
                        trainTestRatio: Double
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

case class TreesConfig(
                        impurity: String,
                        maxBins: Int,
                        maxDepth: Int,
                        minInfoGain: Double,
                        minInstancesPerNode: Int
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

case class XGBoostModelsWithResults(
                                   modelHyperParams: XGBoostConfig,
                                   model: Any,
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

case class TreesModelsWithResults(
                                 modelHyperParams: TreesConfig,
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

case class MLPCConfig(
                       layers: Array[Int],
                       maxIter: Int,
                       solver: String,
                       stepSize: Double,
                       tol: Double
                     )

case class MLPCModelsWithResults(
                                  modelHyperParams: MLPCConfig,
                                  model: MultilayerPerceptronClassificationModel,
                                  score: Double,
                                  evalMetrics: Map[String, Double],
                                  generation: Int
                                )

case class StaticModelConfig(
                              labelColumn: String,
                              featuresColumn: String
                            )

case class GenericModelReturn(
                               hyperParams: Map[String, Any],
                               model: Any,
                               score: Double,
                               metrics: Map[String, Double],
                               generation: Int
                             )

case class GenerationalReport(
                               modelFamily: String,
                               modelType: String,
                               generation: Int,
                               generationMeanScore: Double,
                               generationStddevScore: Double
                             )

case class TreeSplitReport(
                          decisionText: String,
                          featureImportances: DataFrame,
                          model: Any
                          )

case class DataPrepReturn(
                          outputData: DataFrame,
                          fieldListing: Array[String]
                         )

case class OutlierFilteringReturn(
                                 outputData: DataFrame,
                                 fieldRemovalMap: Map[String, (Double, String)]
                                 )

sealed trait Output{
 def modelReport: Array[GenericModelReturn]
 def generationReport: Array[GenerationalReport]
 def modelReportDataFrame: DataFrame
 def generationReportDataFrame: DataFrame
}

abstract case class AutomationOutput() extends Output

abstract case class TunerOutput(rawData: DataFrame, modelSelection:String) extends Output

abstract case class PredictionOutput(dataWithPredictions: DataFrame) extends Output

abstract case class FeatureImportanceOutput(featureImportances: DataFrame) extends Output

abstract case class FeatureImportancePredictionOutput(featureImportances: DataFrame,
                                                      predictionData: DataFrame) extends Output

abstract case class ConfusionOutput(predictionData: DataFrame, confusionData: DataFrame) extends Output

sealed trait ModelType[A, B]

final case class ClassiferType[A, B](a: A) extends ModelType[A, B]

final case class RegressorType[A, B](b: B) extends ModelType[A, B]


