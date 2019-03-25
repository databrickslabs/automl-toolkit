package com.databricks.spark.automatedml.model.tools

import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
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
    require(List("RandomForest", "LinearRegression").contains(value), s"Model type '$value' is not supported for " +
      s"post-run optimization.")
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
    vectorFields +: _numericBoundaries.keys.toArray

    val stringIndexerBuffer = new ArrayBuffer[StringIndexer]

    // Get the string type fields from the Dataframe to StringIndex them
    _stringBoundaries.foreach{ x =>

      val indexedName = s"${x._1}_si"

      val stringIndexer = new StringIndexer()
        .setInputCol(x._1)
        .setOutputCol(indexedName)

      stringIndexerBuffer += stringIndexer

      vectorFields += indexedName
      pipelineBuffer += stringIndexer
    }

    // Build the vector
    val vectorizer = new VectorAssembler()
      .setInputCols(vectorFields.result.toArray)
      .setOutputCol("features")

    pipelineBuffer += vectorizer

    val model = _modelType match {
      case "RandomForest" => new RandomForestRegressor()
      case "LinearRegression" => new LinearRegression()
    }

    model.setLabelCol("score")
      .setFeaturesCol("features")

    pipelineBuffer += model

    // Build the pipeline
    val fullPipeline = new Pipeline()
      .setStages(pipelineBuffer.result.toArray)

    // Fit the model pipeline and return it
    fullPipeline.fit(modelResults)
  }

}
