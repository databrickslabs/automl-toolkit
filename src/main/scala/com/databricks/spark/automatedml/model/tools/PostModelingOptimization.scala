package com.databricks.spark.automatedml.model.tools

import com.databricks.spark.automatedml.model.tools.structures.ModelConfigGenerators
import com.databricks.spark.automatedml.params.{Defaults, RandomForestConfig}

class PostModelingOptimization extends Defaults with ModelConfigGenerators {

  var _modelFamily = ""
  var _modelType = ""
  var _hyperParameterSpaceCount = 100000
  var _numericBoundaries: Map[String, (Double, Double)] = _
  var _stringBoundaries: Map[String, List[String]] = _


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


  def getModelFamily: String = _modelFamily
  def getModelType: String = _modelType
  def getHyperParameterSpaceCount: Int = _hyperParameterSpaceCount


  // TODO: method for generating the hyper param search space

  def generateRandomForestSearchSpace(): Array[RandomForestConfig] = {

    val calculatedPermutationValue = getPermutationCounts(_hyperParameterSpaceCount, _numericBoundaries.size)

  }





}
