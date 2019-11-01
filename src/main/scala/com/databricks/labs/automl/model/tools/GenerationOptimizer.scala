package com.databricks.labs.automl.model.tools

import com.databricks.labs.automl.exceptions.ModelingTypeException
import com.databricks.labs.automl.model.tools.structures._
import com.databricks.labs.automl.params._
import com.databricks.labs.automl.utils.SparkSessionWrapper
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.feature.{
  MaxAbsScaler,
  StringIndexer,
  VectorAssembler
}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

case class LayerConfig(layers: Int, hiddenLayers: Int)

case class MLPCExtractConfig(layers: Int,
                             maxIter: Int,
                             solver: String,
                             stepSize: Double,
                             tolerance: Double,
                             hiddenLayerSize: Int)

case class FieldTypes(numericHyperParams: Array[String],
                      stringHyperParams: Array[String],
                      allHyperParams: Array[String])

object ModelTypes extends Enumeration {
  type ModelTypes = Value
  val Trees, GBT, LinearRegressor, LogisticRegression, MLPC, NaiveBayes,
  RandomForest, SVM, XGBoost, LightGBM = Value
}

object RegressorTypes extends Enumeration {
  type RegressorTypes = Value
  val RF, LR, XG = Value
}

object OptimizationTypes extends Enumeration {
  type OptimizationTypes = Value
  val Minimize, Maximize = Value
}

trait GenerationOptimizerBase extends SparkSessionWrapper {

  import com.databricks.labs.automl.model.tools.ModelTypes._
  import com.databricks.labs.automl.model.tools.OptimizationTypes._
  import com.databricks.labs.automl.model.tools.RegressorTypes._

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
      case "LinearRegression"   => LinearRegressor
      case "LogisticRegression" => LogisticRegression
      case "MLPC"               => MLPC
      case "NaiveBayes"         => NaiveBayes
      case "RandomForest"       => RandomForest
      case "SVM"                => SVM
      case "XGBoost"            => XGBoost
      case "LightGBM"           => LightGBM
      case _ =>
        throw ModelingTypeException(
          value,
          ModelTypes.values.map(_.toString).toArray
        )
    }

  }

  def enumerateRegressorType(value: String): RegressorTypes = {
    value match {
      case "RandomForest"     => RF
      case "LinearRegression" => LR
      case "XGBoost"          => XG
      case _ =>
        throw ModelingTypeException(
          value,
          RegressorTypes.values.map(_.toString).toArray
        )
    }
  }

  def enumerateOptimizationType(value: String): OptimizationTypes = {
    value match {
      case "minimize" => Minimize
      case "maximize" => Maximize
      case _ =>
        throw ModelingTypeException(value, Array("minimize", "maximize"))
    }
  }

  def convertConfigToDF[A](modelType: ModelTypes, config: Array[A])(
    implicit c: ClassTag[A]
  ): DataFrame = {

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
      case LinearRegressor =>
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
      case LightGBM =>
        val conf = config.asInstanceOf[Array[LightGBMModelsWithResults]]
        val report = conf.map(x => {
          val hyperParams = x.modelHyperParams
          LightGBMModelRunReport(
            baggingFraction = hyperParams.baggingFraction,
            baggingFreq = hyperParams.baggingFreq,
            featureFraction = hyperParams.featureFraction,
            learningRate = hyperParams.learningRate,
            maxBin = hyperParams.maxBin,
            maxDepth = hyperParams.maxDepth,
            minSumHessianInLeaf = hyperParams.minSumHessianInLeaf,
            numIterations = hyperParams.numIterations,
            numLeaves = hyperParams.numLeaves,
            boostFromAverage = hyperParams.boostFromAverage,
            lambdaL1 = hyperParams.lambdaL1,
            lambdaL2 = hyperParams.lambdaL2,
            alpha = hyperParams.alpha,
            boostingType = hyperParams.boostingType,
            score = x.score
          )
        })
        spark.createDataFrame(report)
    }
    data
  }

  def convertCandidatesToDF[B](modelType: ModelTypes,
                               candidates: Array[B]): DataFrame = {
    modelType match {
      case Trees =>
        spark.createDataFrame(candidates.asInstanceOf[Array[TreesConfig]])
      case GBT =>
        spark.createDataFrame(candidates.asInstanceOf[Array[GBTConfig]])
      case LinearRegressor =>
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
      case LightGBM =>
        spark.createDataFrame(candidates.asInstanceOf[Array[LightGBMConfig]])
    }
  }

  def fit(df: Dataset[_], pipeline: Pipeline): PipelineModel = {

    pipeline.fit(df)

  }

  def transform(df: Dataset[_], pipeline: PipelineModel): DataFrame = {

    pipeline.transform(df)

  }

}

class GenerationOptimizer[A, B](val modelType: String,
                                val regressorType: String,
                                var history: ArrayBuffer[A],
                                var candidates: Array[B],
                                val optimizationType: String,
                                val candidateCount: Int)
    extends GenerationOptimizerBase {

  import com.databricks.labs.automl.model.tools.OptimizationTypes._
  import com.databricks.labs.automl.model.tools.RegressorTypes._

  final val LABEL_COLUMN: String = "score"
  final val UNSCALED_FEATURE_COLUMN: String = "features"
  final val SCALED_FEATURE_COLUMN: String = "features_scaled"
  final val PREDICTION_COLUMN: String = "predicted_score"
  final val SI_SUFFIX: String = "_si"

  private final val modelEnum = enumerateModelType(modelType)
  private final val regressorEnum = enumerateRegressorType(regressorType)
  private final val optimizationEnum = enumerateOptimizationType(
    optimizationType
  )

  private def extractFieldsToStringIndex(schema: StructType): FieldTypes = {

    val allHyperParams = schema.names.filterNot(LABEL_COLUMN.contains)
    val stringHyperParams = schema
      .filter(_.dataType == StringType)
      .map(_.name)
      .toArray
      .filterNot(LABEL_COLUMN.contains)
    val numericHyperParams =
      allHyperParams.filterNot(stringHyperParams.contains)

    FieldTypes(
      numericHyperParams = numericHyperParams,
      stringHyperParams = stringHyperParams,
      allHyperParams = allHyperParams
    )
  }

  private def buildFeaturePipeline(fields: FieldTypes): Pipeline = {

    val stringIndexers = fields.stringHyperParams.map(
      x => new StringIndexer().setInputCol(x).setOutputCol(x + SI_SUFFIX)
    )
    val vectorNames = fields.stringHyperParams.map(_ + SI_SUFFIX) ++ fields.numericHyperParams

    val vectorAssembler = new VectorAssembler()
      .setInputCols(vectorNames)
      .setOutputCol(UNSCALED_FEATURE_COLUMN)

    val scaler = new MaxAbsScaler()
      .setInputCol(UNSCALED_FEATURE_COLUMN)
      .setOutputCol(SCALED_FEATURE_COLUMN)

    val regressor = regressorEnum match {
      case LR => new LinearRegression().setPredictionCol(PREDICTION_COLUMN)
      case RF => new RandomForestRegressor().setPredictionCol(PREDICTION_COLUMN)
      case XG =>
        new XGBoostRegressor()
          .setMissing(0.0f)
          .setPredictionCol(PREDICTION_COLUMN)
    }

    regressor.setLabelCol(LABEL_COLUMN).setFeaturesCol(SCALED_FEATURE_COLUMN)

    new Pipeline()
      .setStages(stringIndexers :+ vectorAssembler :+ scaler :+ regressor)

  }

  private def sortRestrict(df: DataFrame, limit: Int): DataFrame = {
    optimizationEnum match {
      case Maximize => df.orderBy(col(PREDICTION_COLUMN).desc).limit(limit)
      case Minimize => df.orderBy(col(PREDICTION_COLUMN).asc).limit(limit)
    }

  }

  private def evaluateCandidates()(implicit c: ClassTag[A]): DataFrame = {

    val historyDF = convertConfigToDF(modelEnum, history.toArray)

    val historyFields = extractFieldsToStringIndex(historyDF.schema)

    val candidateDF = convertCandidatesToDF(modelEnum, candidates)

    val candidateFields = extractFieldsToStringIndex(candidateDF.schema)

    val pipeline = buildFeaturePipeline(historyFields)

    val model = fit(historyDF, pipeline)

    val prediction = transform(candidateDF, model)

    sortRestrict(prediction, candidateCount)

  }

  def generateRandomForestCandidates()(
    implicit c: ClassTag[A]
  ): Array[RandomForestConfig] = {

    val candidates = evaluateCandidates()
    candidates
      .collect()
      .map(
        x =>
          RandomForestConfig(
            numTrees = x.getAs[Int]("numTrees"),
            impurity = x.getAs[String]("impurity"),
            maxBins = x.getAs[Int]("maxBins"),
            maxDepth = x.getAs[Int]("maxDepth"),
            minInfoGain = x.getAs[Double]("minInfoGain"),
            subSamplingRate = x.getAs[Double]("subSamplingRate"),
            featureSubsetStrategy = x.getAs[String]("featureSubsetStrategy")
        )
      )
  }

  def generateDecisionTreesCandidates()(
    implicit c: ClassTag[A]
  ): Array[TreesConfig] = {

    val candidates = evaluateCandidates()
    candidates
      .collect()
      .map(
        x =>
          TreesConfig(
            impurity = x.getAs[String]("impurity"),
            maxBins = x.getAs[Int]("maxBins"),
            maxDepth = x.getAs[Int]("maxDepth"),
            minInfoGain = x.getAs[Double]("minInfoGain"),
            minInstancesPerNode = x.getAs[Int]("minInstancesPerNode")
        )
      )
  }

  def generateGBTCandidates()(implicit c: ClassTag[A]): Array[GBTConfig] = {

    val candidates = evaluateCandidates()
    candidates
      .collect()
      .map(
        x =>
          GBTConfig(
            impurity = x.getAs[String]("impurity"),
            lossType = x.getAs[String]("lossType"),
            maxBins = x.getAs[Int]("maxBins"),
            maxDepth = x.getAs[Int]("maxDepth"),
            maxIter = x.getAs[Int]("maxIter"),
            minInfoGain = x.getAs[Double]("minInfoGain"),
            minInstancesPerNode = x.getAs[Int]("minInstancesPerNode"),
            stepSize = x.getAs[Double]("stepSize")
        )
      )
  }

  def generateLinearRegressionCandidates()(
    implicit c: ClassTag[A]
  ): Array[LinearRegressionConfig] = {

    val candidates = evaluateCandidates()
    candidates
      .collect()
      .map(
        x =>
          LinearRegressionConfig(
            elasticNetParams = x.getAs[Double]("elasticNetParams"),
            fitIntercept = x.getAs[Boolean]("fitIntercept"),
            loss = x.getAs[String]("loss"),
            maxIter = x.getAs[Int]("maxIter"),
            regParam = x.getAs[Double]("regParam"),
            standardization = x.getAs[Boolean]("standardization"),
            tolerance = x.getAs[Double]("tolerance")
        )
      )
  }

  def generateLogisticRegressionCandidates()(
    implicit c: ClassTag[A]
  ): Array[LogisticRegressionConfig] = {

    val candidates = evaluateCandidates()
    candidates
      .collect()
      .map(
        x =>
          LogisticRegressionConfig(
            elasticNetParams = x.getAs[Double]("elasticNetParams"),
            fitIntercept = x.getAs[Boolean]("fitIntercept"),
            maxIter = x.getAs[Int]("maxIter"),
            regParam = x.getAs[Double]("regParam"),
            standardization = x.getAs[Boolean]("standardization"),
            tolerance = x.getAs[Double]("tolerance")
        )
      )

  }

  def generateSVMCandidates()(implicit c: ClassTag[A]): Array[SVMConfig] = {

    val candidates = evaluateCandidates()
    candidates
      .collect()
      .map(
        x =>
          SVMConfig(
            fitIntercept = x.getAs[Boolean]("fitIntercept"),
            maxIter = x.getAs[Int]("maxIter"),
            regParam = x.getAs[Double]("regParam"),
            standardization = x.getAs[Boolean]("standardization"),
            tolerance = x.getAs[Double]("tolerance")
        )
      )
  }

  def generateXGBoostCandidates()(
    implicit c: ClassTag[A]
  ): Array[XGBoostConfig] = {

    val candidates = evaluateCandidates()
    candidates
      .collect()
      .map(
        x =>
          XGBoostConfig(
            alpha = x.getAs[Double]("alpha"),
            eta = x.getAs[Double]("eta"),
            gamma = x.getAs[Double]("gamma"),
            lambda = x.getAs[Double]("lambda"),
            maxDepth = x.getAs[Int]("maxDepth"),
            subSample = x.getAs[Double]("subSample"),
            minChildWeight = x.getAs[Double]("minChildWeight"),
            numRound = x.getAs[Int]("numRound"),
            maxBins = x.getAs[Int]("maxBins"),
            trainTestRatio = x.getAs[Double]("trainTestRatio")
        )
      )

  }

  def generateLightGBMCandidates()(
    implicit c: ClassTag[A]
  ): Array[LightGBMConfig] = {
    val candidates = evaluateCandidates()
    candidates
      .collect()
      .map(
        x =>
          LightGBMConfig(
            baggingFraction = x.getAs[Double]("baggingFraction"),
            baggingFreq = x.getAs[Int]("baggingFreq"),
            featureFraction = x.getAs[Double]("featureFraction"),
            learningRate = x.getAs[Double]("learningRate"),
            maxBin = x.getAs[Int]("maxBin"),
            maxDepth = x.getAs[Int]("maxDepth"),
            minSumHessianInLeaf = x.getAs[Double]("minSumHessianInLeaf"),
            numIterations = x.getAs[Int]("numIterations"),
            numLeaves = x.getAs[Int]("numLeaves"),
            boostFromAverage = x.getAs[Boolean]("boostFromAverage"),
            lambdaL1 = x.getAs[Double]("lambdaL1"),
            lambdaL2 = x.getAs[Double]("lambdaL2"),
            alpha = x.getAs[Double]("alpha"),
            boostingType = x.getAs[String]("boostingType")
        )
      )
  }

  def generateMLPCCandidates(inputFeatures: Int, distinctClasses: Int)(
    implicit c: ClassTag[A]
  ): Array[MLPCConfig] = {

    val candidates = evaluateCandidates()
    candidates
      .collect()
      .map(x => {

        val layers = mlpcLayerGenerator(
          inputFeatures,
          distinctClasses,
          x.getAs[Int]("layers"),
          x.getAs[Int]("hiddenLayersSize")
        )

        MLPCConfig(
          layers = layers,
          maxIter = x.getAs[Int]("maxIter"),
          solver = x.getAs[String]("solver"),
          stepSize = x.getAs[Double]("stepSize"),
          tolerance = x.getAs[Double]("tolerance")
        )
      })

  }

}

object GenerationOptimizer {

  def randomForestCandidates[A, B](
    modelType: String,
    regressorType: String,
    history: ArrayBuffer[A],
    candidates: Array[B],
    optimizationType: String,
    candidateCount: Int
  )(implicit c: ClassTag[A]): Array[RandomForestConfig] =
    new GenerationOptimizer(
      modelType,
      regressorType,
      history,
      candidates,
      optimizationType,
      candidateCount
    ).generateRandomForestCandidates()

  def decisionTreesCandidates[A, B](
    modelType: String,
    regressorType: String,
    history: ArrayBuffer[A],
    candidates: Array[B],
    optimizationType: String,
    candidateCount: Int
  )(implicit c: ClassTag[A]): Array[TreesConfig] =
    new GenerationOptimizer(
      modelType,
      regressorType,
      history,
      candidates,
      optimizationType,
      candidateCount
    ).generateDecisionTreesCandidates()

  def gbtCandidates[A, B](
    modelType: String,
    regressorType: String,
    history: ArrayBuffer[A],
    candidates: Array[B],
    optimizationType: String,
    candidateCount: Int
  )(implicit c: ClassTag[A]): Array[GBTConfig] =
    new GenerationOptimizer(
      modelType,
      regressorType,
      history,
      candidates,
      optimizationType,
      candidateCount
    ).generateGBTCandidates()

  def linearRegressionCandidates[A, B](
    modelType: String,
    regressorType: String,
    history: ArrayBuffer[A],
    candidates: Array[B],
    optimizationType: String,
    candidateCount: Int
  )(implicit c: ClassTag[A]): Array[LinearRegressionConfig] =
    new GenerationOptimizer(
      modelType,
      regressorType,
      history,
      candidates,
      optimizationType,
      candidateCount
    ).generateLinearRegressionCandidates()

  def logisticRegressionCandidates[A, B](
    modelType: String,
    regressorType: String,
    history: ArrayBuffer[A],
    candidates: Array[B],
    optimizationType: String,
    candidateCount: Int
  )(implicit c: ClassTag[A]): Array[LogisticRegressionConfig] =
    new GenerationOptimizer(
      modelType,
      regressorType,
      history,
      candidates,
      optimizationType,
      candidateCount
    ).generateLogisticRegressionCandidates()

  def svmCandidates[A, B](
    modelType: String,
    regressorType: String,
    history: ArrayBuffer[A],
    candidates: Array[B],
    optimizationType: String,
    candidateCount: Int
  )(implicit c: ClassTag[A]): Array[SVMConfig] =
    new GenerationOptimizer(
      modelType,
      regressorType,
      history,
      candidates,
      optimizationType,
      candidateCount
    ).generateSVMCandidates()

  def xgBoostCandidates[A, B](
    modelType: String,
    regressorType: String,
    history: ArrayBuffer[A],
    candidates: Array[B],
    optimizationType: String,
    candidateCount: Int
  )(implicit c: ClassTag[A]): Array[XGBoostConfig] =
    new GenerationOptimizer(
      modelType,
      regressorType,
      history,
      candidates,
      optimizationType,
      candidateCount
    ).generateXGBoostCandidates()

  def lightGBMCandidates[A, B](
    modelType: String,
    regressorType: String,
    history: ArrayBuffer[A],
    candidates: Array[B],
    optimizationType: String,
    candidateCount: Int
  )(implicit c: ClassTag[A]): Array[LightGBMConfig] = {
    new GenerationOptimizer(
      modelType,
      regressorType,
      history,
      candidates,
      optimizationType,
      candidateCount
    ).generateLightGBMCandidates()
  }

  def mlpcCandidates[A, B](
    modelType: String,
    regressorType: String,
    history: ArrayBuffer[A],
    candidates: Array[B],
    optimizationType: String,
    candidateCount: Int,
    inputFeatures: Int,
    distinctClasses: Int
  )(implicit c: ClassTag[A]): Array[MLPCConfig] =
    new GenerationOptimizer(
      modelType,
      regressorType,
      history,
      candidates,
      optimizationType,
      candidateCount
    ).generateMLPCCandidates(inputFeatures, distinctClasses)

}
