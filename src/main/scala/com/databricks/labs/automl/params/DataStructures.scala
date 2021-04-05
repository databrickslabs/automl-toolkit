package com.databricks.labs.automl.params

import com.databricks.labs.automl.tracking.MLFlowReportStructure
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.DataFrame

case class PearsonPayload(fieldName: String,
                          pvalue: Double,
                          degreesFreedom: Int,
                          pearsonStat: Double)

case class FeatureCorrelationStats(leftCol: String,
                                   rightCol: String,
                                   correlation: Double)

case class FilterData(field: String, uniqueValues: Long)

case class ManualFilters(field: String, threshold: Double)

// Marker trait for all tuner input configs
trait TunerConfigBase extends Product {}

case class XGBoostConfig(alpha: Double,
                         eta: Double,
                         gamma: Double,
                         lambda: Double,
                         maxDepth: Int,
                         subSample: Double,
                         minChildWeight: Double,
                         numRound: Int,
                         maxBins: Int,
                         trainTestRatio: Double) extends TunerConfigBase

case class RandomForestConfig(numTrees: Int,
                              impurity: String,
                              maxBins: Int,
                              maxDepth: Int,
                              minInfoGain: Double,
                              subSamplingRate: Double,
                              featureSubsetStrategy: String) extends TunerConfigBase

case class TreesConfig(impurity: String,
                       maxBins: Int,
                       maxDepth: Int,
                       minInfoGain: Double,
                       minInstancesPerNode: Int) extends TunerConfigBase

case class GBTConfig(impurity: String,
                     lossType: String,
                     maxBins: Int,
                     maxDepth: Int,
                     maxIter: Int,
                     minInfoGain: Double,
                     minInstancesPerNode: Int,
                     stepSize: Double) extends TunerConfigBase

case class LogisticRegressionConfig(elasticNetParams: Double,
                                    fitIntercept: Boolean,
                                    maxIter: Int,
                                    regParam: Double,
                                    standardization: Boolean,
                                    tolerance: Double) extends TunerConfigBase

case class LinearRegressionConfig(elasticNetParams: Double,
                                  fitIntercept: Boolean,
                                  loss: String,
                                  maxIter: Int,
                                  regParam: Double,
                                  standardization: Boolean,
                                  tolerance: Double) extends TunerConfigBase

// Market trait for tuner results
trait TunerOutputWithResults[A, B]{
  def modelHyperParams: A
  def model: B
  def score: Double
  def evalMetrics: Map[String, Double]
  def generation: Int
}

case class LinearRegressionModelsWithResults(override val modelHyperParams: LinearRegressionConfig,
                                             override val model: LinearRegressionModel,
                                             override val score: Double,
                                             override val evalMetrics: Map[String, Double],
                                             override val generation: Int)
  extends TunerOutputWithResults[LinearRegressionConfig, LinearRegressionModel]

case class LogisticRegressionModelsWithResults(override val modelHyperParams: LogisticRegressionConfig,
                                               override val model: LogisticRegressionModel,
                                               override val score: Double,
                                               override val evalMetrics: Map[String, Double],
                                               override val generation: Int)
  extends TunerOutputWithResults[LogisticRegressionConfig, LogisticRegressionModel]


case class XGBoostModelsWithResults(override val modelHyperParams: XGBoostConfig,
                                    override val model: Any,
                                    override val score: Double,
                                    override val evalMetrics: Map[String, Double],
                                    override val generation: Int)
  extends TunerOutputWithResults[XGBoostConfig, Any]

case class RandomForestModelsWithResults(override val modelHyperParams: RandomForestConfig,
                                         override val model: Any,
                                         override val score: Double,
                                         override val evalMetrics: Map[String, Double],
                                         override val generation: Int)
  extends TunerOutputWithResults[RandomForestConfig, Any]

case class TreesModelsWithResults(override val modelHyperParams: TreesConfig,
                                  override val model: Any,
                                  override val score: Double,
                                  override val evalMetrics: Map[String, Double],
                                  override val generation: Int)
  extends TunerOutputWithResults[TreesConfig, Any]

case class GBTModelsWithResults(override val modelHyperParams: GBTConfig,
                                override val model: Any,
                                override val score: Double,
                                override val evalMetrics: Map[String, Double],
                                override val generation: Int)
  extends TunerOutputWithResults[GBTConfig, Any]

case class SVMConfig(fitIntercept: Boolean,
                     maxIter: Int,
                     regParam: Double,
                     standardization: Boolean,
                     tolerance: Double) extends TunerConfigBase

case class SVMModelsWithResults(override val modelHyperParams: SVMConfig,
                                override val model: LinearSVCModel,
                                override val score: Double,
                                override val evalMetrics: Map[String, Double],
                                override val generation: Int)
  extends TunerOutputWithResults[SVMConfig, LinearSVCModel]

case class MLPCConfig(layers: Array[Int],
                      maxIter: Int,
                      solver: String,
                      stepSize: Double,
                      tolerance: Double) extends TunerConfigBase

case class MLPCModelsWithResults(override val modelHyperParams: MLPCConfig,
                                 override val model: MultilayerPerceptronClassificationModel,
                                 override val score: Double,
                                 override val evalMetrics: Map[String, Double],
                                 override val generation: Int)
  extends TunerOutputWithResults[MLPCConfig, MultilayerPerceptronClassificationModel]

case class NaiveBayesConfig(modelType: String,
                            smoothing: Double,
                            thresholds: Boolean) extends TunerConfigBase

case class NaiveBayesModelsWithResults(override val modelHyperParams: NaiveBayesConfig,
                                       override val model: NaiveBayesModel,
                                       override val score: Double,
                                       override val evalMetrics: Map[String, Double],
                                       override val generation: Int)
  extends TunerOutputWithResults[NaiveBayesConfig, NaiveBayesModel]

case class LightGBMConfig(baggingFraction: Double,
                          baggingFreq: Int,
                          featureFraction: Double,
                          learningRate: Double,
                          maxBin: Int,
                          maxDepth: Int,
                          minSumHessianInLeaf: Double,
                          numIterations: Int,
                          numLeaves: Int,
                          boostFromAverage: Boolean,
                          lambdaL1: Double,
                          lambdaL2: Double,
                          alpha: Double,
                          boostingType: String) extends TunerConfigBase

case class LightGBMModelsWithResults(override val modelHyperParams: LightGBMConfig,
                                     override val model: Any,
                                     override val score: Double,
                                     override val evalMetrics: Map[String, Double],
                                     override val generation: Int)
  extends TunerOutputWithResults[LightGBMConfig, Any]

case class StaticModelConfig(labelColumn: String, featuresColumn: String)

case class GenericModelReturn(hyperParams: Map[String, Any],
                              model: Any,
                              score: Double,
                              metrics: Map[String, Double],
                              generation: Int)

case class GroupedModelReturn(modelFamily: String,
                              hyperParams: Map[String, Any],
                              model: Any,
                              score: Double,
                              metrics: Map[String, Double],
                              generation: Int)

case class GenerationalReport(modelFamily: String,
                              modelType: String,
                              generation: Int,
                              generationMeanScore: Double,
                              generationStddevScore: Double)

case class FeatureImportanceReturn(modelPayload: RandomForestModelsWithResults,
                                   data: DataFrame,
                                   fields: Array[String],
                                   modelType: String)

case class TreeSplitReport(decisionText: String,
                           featureImportances: DataFrame,
                           model: Any)

case class DataPrepReturn(outputData: DataFrame, fieldListing: Array[String])

case class DataGeneration(data: DataFrame,
                          fields: Array[String],
                          modelType: String)

case class OutlierFilteringReturn(
  outputData: DataFrame,
  fieldRemovalMap: Map[String, (Double, String)]
)

sealed trait Output {
  def modelReport: Array[GenericModelReturn]
  def generationReport: Array[GenerationalReport]
  def modelReportDataFrame: DataFrame
  def generationReportDataFrame: DataFrame
}

abstract case class AutomationOutput(mlFlowOutput: MLFlowReportStructure)
    extends Output

abstract case class TunerOutput(rawData: DataFrame,
                                modelSelection: String,
                                mlFlowOutput: MLFlowReportStructure)
    extends Output

abstract case class PredictionOutput(dataWithPredictions: DataFrame,
                                     mlFlowOutput: MLFlowReportStructure)
    extends Output

abstract case class FeatureImportanceOutput(featureImportances: DataFrame,
                                            mlFlowOutput: MLFlowReportStructure)
    extends Output

abstract case class FeatureImportancePredictionOutput(
  featureImportances: DataFrame,
  predictionData: DataFrame,
  mlFlowOutput: MLFlowReportStructure
) extends Output

abstract case class ConfusionOutput(predictionData: DataFrame,
                                    confusionData: DataFrame,
                                    mlFlowOutput: MLFlowReportStructure)
    extends Output

abstract case class FamilyOutput(modelType: String,
                                 mlFlowOutput: MLFlowReportStructure)
    extends Output

case class FamilyFinalOutput(modelReport: Array[GroupedModelReturn],
                             generationReport: Array[GenerationalReport],
                             modelReportDataFrame: DataFrame,
                             generationReportDataFrame: DataFrame,
                             mlFlowReport: Array[MLFlowReportStructure])

case class FamilyFinalOutputWithPipeline(
  familyFinalOutput: FamilyFinalOutput,
  bestPipelineModel: Map[String, PipelineModel],
  bestMlFlowRunId: Map[String, String] = Map.empty
)

sealed trait ModelType[A, B]

final case class ClassiferType[A, B](a: A) extends ModelType[A, B]

final case class RegressorType[A, B](b: B) extends ModelType[A, B]
