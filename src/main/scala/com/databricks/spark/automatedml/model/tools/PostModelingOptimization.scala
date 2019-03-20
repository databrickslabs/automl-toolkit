package com.databricks.spark.automatedml.model.tools

import com.databricks.spark.automatedml.model.tools.structures.{ModelConfigGenerators, PermutationConfiguration, RandomForestModelRunReport}
import com.databricks.spark.automatedml.params.{Defaults, GenericModelReturn, RandomForestConfig}
import com.databricks.spark.automatedml.utils.SparkSessionWrapper
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

class PostModelingOptimization extends Defaults with ModelConfigGenerators with SparkSessionWrapper {

  var _modelFamily = ""
  var _modelType = ""
  var _hyperParameterSpaceCount = 100000
  var _numericBoundaries: Map[String, (Double, Double)] = _
  var _stringBoundaries: Map[String, List[String]] = _
  var _seed: Long = 42L


  def setModelFamily(value: String): this.type = {
    require(_supportedModels.contains(value), s"${this.getClass.toString} error! Model Family $value is not supported." +
      s"\n\t Supported families: ${_supportedModels.mkString(", ")}")
    _modelFamily = value
    this
  }

  def setModelType(value: String): this.type = {
    value match {
      case "classifier" => _modelType = value
      case "regressor" => _modelType = value
      case _ => throw new UnsupportedOperationException(s"Model type $value is not supported.")
    }
    this
  }

  def setHyperParameterSpaceCount(value: Int): this.type = {
    value match {
      case x if x > 500000 => println(s"WARNING! HyperParameterSpaceCount value of $x is above 500,000.  " +
        s"This will increase driver memory pressure and run time. Proceed if this is a desired setting only.")
      case y if y > 1000000 => throw new UnsupportedOperationException(s"HyperParameterSpaceCount setting of $y is " +
        s"greater than the allowable maximum of 1,000,000 permutations")
    }
    _hyperParameterSpaceCount = value
    this
  }

  def setNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    _numericBoundaries = value
    this
  }

  def setStringBoundaries(value: Map[String, List[String]]): this.type = {
    _stringBoundaries = value
    this
  }

  def setSeed(value: Long): this.type = {
    _seed = value
    this
  }

  def getModelFamily: String = _modelFamily
  def getModelType: String = _modelType
  def getHyperParameterSpaceCount: Int = _hyperParameterSpaceCount
  def getNumericBoundaries: Map[String, (Double, Double)] = _numericBoundaries
  def getStringBoundaries: Map[String, List[String]] = _stringBoundaries
  def getSeed: Long = _seed

  /**
    * Generates an array of RandomForestConfig hyper parameters to meet the configured target size
    * @return a distinct array of RandomForestConfig's
    */
  def generateRandomForestSearchSpace(): Array[RandomForestConfig] = {

    // Get the number of permutations to create for the continuous numeric boundary search space
    val calculatedPermutationValue = getPermutationCounts(_hyperParameterSpaceCount, _numericBoundaries.size) +
      stringBoundaryPermutationCalculator(_stringBoundaries)

    // Specify the Permutation Configuration
    val permutationConfig = PermutationConfiguration(
      permutationTarget = calculatedPermutationValue,
      numericBoundaries = _numericBoundaries,
      stringBoundaries = _stringBoundaries
    )

    // Generate the Permutations
    val permutationsArray = randomForestPermutationGenerator(permutationConfig, _hyperParameterSpaceCount, _seed)

    permutationsArray.distinct
  }

  def generateRandomForestSearchSpaceAsDataFrame(): DataFrame = {

    spark.createDataFrame(generateRandomForestSearchSpace())

  }

  protected[tools] def randomForestResultMapping(results: Array[GenericModelReturn]): DataFrame = {

    val builder = new ArrayBuffer[RandomForestModelRunReport]()

    results.foreach{ x =>
      val hyperParams = x.hyperParams
      builder += RandomForestModelRunReport(
        numTrees = hyperParams("numTrees").toString.toInt,
        impurity = hyperParams("impurity").toString,
        maxBins = hyperParams("maxBins").toString.toInt,
        maxDepth = hyperParams("maxDepth").toString.toInt,
        minInfoGain = hyperParams("minInfoGain").toString.toDouble,
        subSamplingRate = hyperParams("subSamplingRate").toString.toDouble,
        featureSubsetStrategy = hyperParams("featureSubsetStrategy").toString,
        score = x.score
      )
    }
    spark.createDataFrame(builder.result.toArray)
  }

  def randomForestPrediction(modelingResults: Array[GenericModelReturn], modelType: String, topPredictions: Int):
  Array[RandomForestConfig] = {

    val inferenceDataSet = randomForestResultMapping(modelingResults)

    val impuritySi = new StringIndexer()
      .setInputCol("impurity")
      .setOutputCol("impurity_si")

    val featureSubsetStrategySi = new StringIndexer()
      .setInputCol("featureSubsetStrategy")
      .setOutputCol("featureSubsetStrategy_si")

    val vectorizer = new VectorAssembler()
      .setInputCols(Array("numTrees", "impurity_si", "maxBins", "maxDepth", "minInfoGain",
      "subSamplingRate", "featureSubsetStrategy_si"))
      .setOutputCol("features")

    //TODO: set some solid hyper parameters for these models to ensure that the results are relatively accurate.
    val model = modelType match {
      case "RandomForest" => new RandomForestRegressor()
      case "LinearRegression" => new LinearRegression()
    }

    model.setLabelCol("score")
      .setFeaturesCol("features")

    val modelPipeline = new Pipeline()
      .setStages(Array(impuritySi, featureSubsetStrategySi, vectorizer, model))

    val fittedPipeline = modelPipeline.fit(inferenceDataSet)

    val fullSearchSpaceDataSet = generateRandomForestSearchSpaceAsDataFrame()

    fittedPipeline.transform(fullSearchSpaceDataSet)
      .orderBy(col("prediction").desc)
      .limit(topPredictions)
      .collect()
      .asInstanceOf[Array[RandomForestConfig]]

  }






}
