package com.databricks.spark.automatedml

import com.databricks.spark.automatedml.model.RandomForestTuner
import com.databricks.spark.automatedml.params.{GBTModelsWithResults, MLPCModelsWithResults, RandomForestModelsWithResults}
import com.databricks.spark.automatedml.utils.SparkSessionWrapper
import org.apache.spark.sql.DataFrame



trait Defaults {

  final val _supportedModels: Array[String] = Array(
    "GBT",
    "RandomForest",
    "LinearRegression",
    "LogisticRegression",
    "MLPC",
    "SVM"
  )

  val rfDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "numTrees" -> Tuple2(50.0, 1000.0),
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "subSamplingRate" -> Tuple2(0.5, 1.0)
  )

  val rfDefaultStringBoundaries = Map(
    "impurity" -> List("gini", "entropy"),
    "featureSubsetStrategy" -> List("all", "sqrt", "log2", "onethird")
  )

  val mlpcDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "layers" -> Tuple2(1.0, 10.0),
    "maxIter" -> Tuple2(100.0, 10000.0),
    "stepSize" -> Tuple2(0.01, 1.0),
    "tol" -> Tuple2(1E-9, 1E-5),
    "hiddenLayerSizeAdjust" -> Tuple2(0.0, 50.0)
  )

  val mlpcDefaultStringBoundaries: Map[String, List[String]] = Map(
    "solver" -> List("gd", "l-bfgs")
  )

  val scoringDefaultClassifier = "f1"
  val scoringDefaultRegressor = "rmse"

}

case class MainConfig(
                     modelType: String = "RandomForest",
                     df: DataFrame,
                     labelCol: String = "label",
                     featuresCol: String = "features",
                     numericBoundaries: Option[Map[String, (Double, Double)]]=None,
                     stringBoundaries: Option[Map[String, List[String]]]=None,
                     scoringMetric: Option[String]=None

                     )

case class ModelingConfig(
                         labelCol: String,
                         featuresCol: String,
                         numericBoundaries: Map[String, (Double, Double)],
                         stringBoundaries: Map[String, List[String]],
                         scoringMetric: String
                         )


class Automation(config: MainConfig) extends Defaults with SparkSessionWrapper{

  require(_supportedModels.contains(config.modelType))


  private val _modelParams = ModelingConfig(
    labelCol = config.labelCol,
    featuresCol = config.featuresCol,
    numericBoundaries = config.modelType match {
      case "RandomForest" => config.numericBoundaries.getOrElse(rfDefaultNumBoundaries)
      case "MLPC" => config.numericBoundaries.getOrElse(mlpcDefaultNumBoundaries)
    },
    stringBoundaries = config.modelType match {
      case "RandomForest" => config.stringBoundaries.getOrElse(rfDefaultStringBoundaries)
      case "MLPC" => config.stringBoundaries.getOrElse(mlpcDefaultStringBoundaries)
    },
    scoringMetric = config.modelType match {
      case "RandomForest" => config.scoringMetric.getOrElse(scoringDefaultClassifier)
      case "MLPC" => config.scoringMetric.getOrElse(scoringDefaultClassifier)
    }
  )



  def getModelConfig: ModelingConfig = _modelParams


  // TODO: write a different method execution for each of the main model types.  DUH.

//  def runRandomForest(): (Array[RandomForestModelsWithResults], DataFrame) = {
//
//        new RandomForestTuner(config.df, "classifier", _modelParams)
//          .evolveWithScoringDF()
//
//  }


  


}




/**
  * Import Config (which elements to do) and their settings
  * Run pipeline
  * Extract Fields
  * Filter / Sanitize
  * Run chosen Model
  * Extract Best
  * Run Report for Feature Importances
  * Run Report for Decision Tree
  * Export Reports + Importances + Models + Final DataFrame
  */