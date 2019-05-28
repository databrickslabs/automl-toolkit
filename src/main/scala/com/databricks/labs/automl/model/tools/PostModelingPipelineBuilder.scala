package com.databricks.labs.automl.model.tools

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class PostModelingPipelineBuilder(modelResults: DataFrame) {

  var _numericBoundaries: Map[String, (Double, Double)] = _
  var _stringBoundaries: Map[String, List[String]] = _
  var _modelType: String = _

  def setNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    _numericBoundaries = value
    this
  }

  def setStringBoundaries(value: Map[String, List[String]]): this.type = {
    _stringBoundaries = value
    this
  }

  def setModelType(value: String): this.type = {
    require(
      List("RandomForest", "LinearRegression", "XGBoost").contains(value),
      s"Model type '$value' is not supported for " +
        s"post-run optimization."
    )
    _modelType = value
    this
  }

  def getNumericBoundaries: Map[String, (Double, Double)] = _numericBoundaries
  def getStringBoundaries: Map[String, List[String]] = _stringBoundaries
  def getModelType: String = _modelType

  protected[tools] def regressionModelForPermutationTest(): PipelineModel = {

    val vectorFields = new ArrayBuffer[String]
    val pipelineBuffer = new ArrayBuffer[PipelineStage]

    // Insert the Numeric Values directly into the ArrayBuffer for column listings for the vector assembler
    _numericBoundaries.keys.toArray.foreach(x => vectorFields += x)

    // Get the string type fields from the Dataframe to StringIndex them
//    _stringBoundaries.keys.foreach{ x =>
//
//      val indexedName = s"${x}_si"
//
//      val stringIndexer = new StringIndexer()
//        .setInputCol(x)
//        .setOutputCol(indexedName)
//        .setHandleInvalid("keep")
//
//      vectorFields += indexedName
//      pipelineBuffer += stringIndexer
//    }

    // Build the vector
    val vectorizer = new VectorAssembler()
      .setInputCols(vectorFields.result.toArray)
      .setOutputCol("features")

    pipelineBuffer += vectorizer

    val model = _modelType match {
      case "RandomForest" =>
        new RandomForestRegressor()
          .setMinInfoGain(1E-8)
          .setNumTrees(600)
          .setMaxDepth(10)
      case "LinearRegression" => new LinearRegression()
      case "XGBoost" =>
        new XGBoostRegressor()
          .setAlpha(0.5)
          .setEta(0.25)
          .setGamma(3.0)
          .setLambda(10.0)
          .setMaxBins(200)
          .setMaxDepth(10)
          .setMinChildWeight(3.0)
          .setNumRound(10)
    }

    model
      .setLabelCol("score")
      .setFeaturesCol("features")

    pipelineBuffer += model

    // Build the pipeline
    val fullPipeline = new Pipeline()
      .setStages(pipelineBuffer.result.toArray)

    // Fit the model pipeline and return it
    fullPipeline.fit(modelResults)
  }

}
