package com.databricks.labs.automl.model.tools

import com.databricks.labs.automl.exceptions.ModelingTypeException
import com.databricks.labs.automl.model.tools.structures.{
  GBTModelRunReport,
  LinearRegressionModelRunReport,
  LogisticRegressionModelRunReport,
  MLPCModelRunReport,
  RandomForestModelRunReport,
  SVMModelRunReport,
  TreesModelRunReport,
  XGBoostModelRunReport
}
import com.databricks.labs.automl.params.{
  GBTConfig,
  GBTModelsWithResults,
  LinearRegressionConfig,
  LinearRegressionModelsWithResults,
  LogisticRegressionConfig,
  LogisticRegressionModelsWithResults,
  MLPCConfig,
  MLPCModelsWithResults,
  RandomForestConfig,
  RandomForestModelsWithResults,
  SVMConfig,
  SVMModelsWithResults,
  TreesConfig,
  TreesModelsWithResults,
  XGBoostConfig,
  XGBoostModelsWithResults
}
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StringType

import scala.collection.mutable.ArrayBuffer

case class LayerConfig(layers: Int, hiddenLayers: Int)

trait GenerationOptimizerBase extends SparkSessionWrapper {

  import com.databricks.labs.automl.model.tools.ModelTypes._

  private def layerExtract(layers: Array[Int]): LayerConfig = {

    val hiddenLayersSizeAdjust =
      if (layers.length > 2) layers(1) - layers(0) else 0
    val layerCount = layers.length - 2

    LayerConfig(layerCount, hiddenLayersSizeAdjust)

  }

  def mlpcLayerGenerator(inputFeatures: Int,
                         distinctClasses: Int,
                         layers: Int,
                         hiddenLayers: Int): Array[Int] = {

    val layerConstruct = new ArrayBuffer[Int]

    layerConstruct += inputFeatures

    (1 to layers).foreach { x =>
      layerConstruct += inputFeatures + layers - x + hiddenLayers
    }
    layerConstruct += distinctClasses
    layerConstruct.result.toArray

  }

  def enumerateModelType(value: String): ModelTypes = {

    value match {
      case "Trees"              => Trees
      case "GBT"                => GBT
      case "LinearRegression"   => LinearRegression
      case "LogisticRegression" => LogisticRegression
      case "MLPC"               => MLPC
      case "NaiveBayes"         => NaiveBayes
      case "RandomForest"       => RandomForest
      case "SVM"                => SVM
      case "XGBoost"            => XGBoost
      case _ =>
        throw ModelingTypeException(
          value,
          ModelTypes.values.map(_.toString).toArray
        )
    }

  }

  def convertConfigToDF[A](modelType: ModelTypes,
                           config: Array[A]): DataFrame = {

    val data = modelType match {
      case Trees =>
        val conf = config.asInstanceOf[Array[TreesModelsWithResults]]
        val report = conf.map(x => {
          val hyperParams = x.modelHyperParams
          TreesModelRunReport(
            impurity = hyperParams.impurity,
            maxBins = hyperParams.maxBins,
            maxDepth = hyperParams.maxDepth,
            minInfoGain = hyperParams.minInfoGain,
            minInstancesPerNode = hyperParams.minInstancesPerNode,
            score = x.score
          )
        })
        spark.createDataFrame(report)
      case GBT =>
        val conf = config.asInstanceOf[Array[GBTModelsWithResults]]
        val report = conf.map(x => {
          val hyperParams = x.modelHyperParams
          GBTModelRunReport(
            impurity = hyperParams.impurity,
            lossType = hyperParams.lossType,
            maxBins = hyperParams.maxBins,
            maxDepth = hyperParams.maxDepth,
            maxIter = hyperParams.maxIter,
            minInfoGain = hyperParams.minInfoGain,
            minInstancesPerNode = hyperParams.minInstancesPerNode,
            stepSize = hyperParams.stepSize,
            score = x.score
          )
        })
        spark.createDataFrame(report)
      case LinearRegression =>
        val conf = config.asInstanceOf[Array[LinearRegressionModelsWithResults]]
        val report = conf.map(x => {
          val hyperParams = x.modelHyperParams
          LinearRegressionModelRunReport(
            elasticNetParams = hyperParams.elasticNetParams,
            fitIntercept = hyperParams.fitIntercept,
            loss = hyperParams.loss,
            maxIter = hyperParams.maxIter,
            regParam = hyperParams.regParam,
            standardization = hyperParams.standardization,
            tolerance = hyperParams.tolerance,
            score = x.score
          )
        })
        spark.createDataFrame(report)
      case LogisticRegression =>
        val conf =
          config.asInstanceOf[Array[LogisticRegressionModelsWithResults]]
        val report = conf.map(x => {
          val hyperParams = x.modelHyperParams
          LogisticRegressionModelRunReport(
            elasticNetParams = hyperParams.elasticNetParams,
            fitIntercept = hyperParams.fitIntercept,
            maxIter = hyperParams.maxIter,
            regParam = hyperParams.regParam,
            standardization = hyperParams.standardization,
            tolerance = hyperParams.tolerance,
            score = x.score
          )
        })
        spark.createDataFrame(report)
      case MLPC =>
        val conf = config.asInstanceOf[Array[MLPCModelsWithResults]]
        val report = conf.map(x => {
          val hyperParams = x.modelHyperParams
          val layers = layerExtract(hyperParams.layers)
          MLPCModelRunReport(
            layers = layers.layers,
            maxIter = hyperParams.maxIter,
            solver = hyperParams.solver,
            stepSize = hyperParams.stepSize,
            tolerance = hyperParams.tolerance,
            hiddenLayerSizeAdjust = layers.hiddenLayers,
            score = x.score
          )
        })
        spark.createDataFrame(report)
      case RandomForest =>
        val conf = config.asInstanceOf[Array[RandomForestModelsWithResults]]
        val report = conf.map(x => {
          val hyperParams = x.modelHyperParams
          RandomForestModelRunReport(
            numTrees = hyperParams.numTrees,
            impurity = hyperParams.impurity,
            maxBins = hyperParams.maxBins,
            maxDepth = hyperParams.maxDepth,
            minInfoGain = hyperParams.minInfoGain,
            subSamplingRate = hyperParams.subSamplingRate,
            featureSubsetStrategy = hyperParams.featureSubsetStrategy,
            score = x.score
          )
        })
        spark.createDataFrame(report)
      case SVM =>
        val conf = config.asInstanceOf[Array[SVMModelsWithResults]]
        val report = conf.map(x => {
          val hyperParams = x.modelHyperParams
          SVMModelRunReport(
            fitIntercept = hyperParams.fitIntercept,
            maxIter = hyperParams.maxIter,
            regParam = hyperParams.regParam,
            standardization = hyperParams.standardization,
            tolerance = hyperParams.tolerance,
            score = x.score
          )
        })
        spark.createDataFrame(report)
      case XGBoost =>
        val conf = config.asInstanceOf[Array[XGBoostModelsWithResults]]
        val report = conf.map(x => {
          val hyperParams = x.modelHyperParams
          XGBoostModelRunReport(
            alpha = hyperParams.alpha,
            eta = hyperParams.eta,
            gamma = hyperParams.gamma,
            lambda = hyperParams.lambda,
            maxDepth = hyperParams.maxDepth,
            subSample = hyperParams.subSample,
            minChildWeight = hyperParams.minChildWeight,
            numRound = hyperParams.numRound,
            maxBins = hyperParams.maxBins,
            trainTestRatio = hyperParams.trainTestRatio,
            score = x.score
          )
        })
        spark.createDataFrame(report)
    }
    data
  }

  def convertCandidatesToDF(modelType: ModelTypes,
                            candidates: Array[AnyRef]): DataFrame = {
    modelType match {
      case Trees =>
        spark.createDataFrame(candidates.asInstanceOf[Array[TreesConfig]])
      case GBT =>
        spark.createDataFrame(candidates.asInstanceOf[Array[GBTConfig]])
      case LinearRegression =>
        spark.createDataFrame(
          candidates.asInstanceOf[Array[LinearRegressionConfig]]
        )
      case LogisticRegression =>
        spark.createDataFrame(
          candidates.asInstanceOf[Array[LogisticRegressionConfig]]
        )
      case MLPC =>
        val conf = candidates.asInstanceOf[Array[MLPCConfig]]
        val adjust = conf.map(x => {
          val layers = layerExtract(x.layers)
          MLPCExtractConfig(
            layers = layers.layers,
            maxIter = x.maxIter,
            solver = x.solver,
            stepSize = x.stepSize,
            tolerance = x.tolerance,
            hiddenLayerSize = layers.hiddenLayers
          )
        })
        spark.createDataFrame(adjust)
      case RandomForest =>
        spark.createDataFrame(
          candidates.asInstanceOf[Array[RandomForestConfig]]
        )
      case SVM =>
        spark.createDataFrame(candidates.asInstanceOf[Array[SVMConfig]])
      case XGBoost =>
        spark.createDataFrame(candidates.asInstanceOf[Array[XGBoostConfig]])
    }
  }

}

object ModelTypes extends Enumeration {
  type ModelTypes = Value
  val Trees, GBT, LinearRegression, LogisticRegression, MLPC, NaiveBayes,
  RandomForest, SVM, XGBoost = Value
}

class GenerationOptimizer[A](val modelType: String, var history: ArrayBuffer[A])
    extends GenerationOptimizerBase {

  private final val modelEnum = enumerateModelType(modelType)

  def evaluateCandidates() = {

    val scoredDF = convertConfigToDF(modelEnum, history.toArray)

    val scoredSchema = scoredDF.schema

    val hyperParamNames = scoredSchema.names.filterNot("score".contains)

    val columnsToStringIndex = scoredSchema.fields.map(
      x =>
        x.dataType match {
          case y if y == StringType => x.name
      }
    )

    val siColumns = columnsToStringIndex.map(x => x + "_si")

    // Build a pipeline to string index the values

    // Vectorize

    // Build a Regressor

    // fit the pipeline on the historical

    // transform the candidates

    // sort, limit

    // convert the config objects to the appropriate config object to return (probably need a method for each model type)

    // don't forget to handle the MLPC re-conversion nonsense to build the layer array

    // The MLPC method needs to have the static values of inputFeatureSize and distinctClasses to function correctly!!!
    // use mlpcLayerGenerator to convert back to MLPCConfig object type for each of the elements of the Dataframe row
    // after filtering!!!!!

  }

}

object GenerationOptimizer {}

case class MLPCExtractConfig(layers: Int,
                             maxIter: Int,
                             solver: String,
                             stepSize: Double,
                             tolerance: Double,
                             hiddenLayerSize: Int)
