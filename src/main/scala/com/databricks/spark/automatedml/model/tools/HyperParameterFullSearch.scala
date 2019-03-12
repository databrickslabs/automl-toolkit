package com.databricks.spark.automatedml.model.tools

import com.databricks.spark.automatedml.params.{Defaults, RandomForestConfig}

import scala.collection.mutable.ArrayBuffer
import util.Random


class HyperParameterFullSearch extends SeedGenerator with Defaults {

  var _modelFamily = ""
  var _modelType = ""
  var _permutationCount = 10

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

  def setPermutationCount(value: Int): this.type = {
    _permutationCount = value
    this
  }

  def getModelFamily: String = _modelFamily
  def getModelType: String = _modelType
  def getPermutationCount: Int = _permutationCount


  private def randomIndexSelection(numericArrays: Array[Array[Double]]): NumericArrayCollection = {

    val arrayRandomHolder = numericArrays.map(x => Random.shuffle(x))

    val randomlySelectedPayload = arrayRandomHolder.map(x => x(0))

    val remainingArrays = arrayRandomHolder.map(x => x.drop(1))

    NumericArrayCollection(randomlySelectedPayload, remainingArrays)

  }

  private def staticIndexSelection(numericArrays: Array[Array[Double]]): NumericArrayCollection = {

    val selectedPayload = numericArrays.map(x => x(0))

    val remainingArrays = numericArrays.map(x => x.drop(1))

    NumericArrayCollection(selectedPayload, remainingArrays)

  }


  private def generateRandomForestHyperParameters(numericBoundaries: Map[String, (Double, Double)],
                                                  stringBoundaries: Map[String, List[String]]): Array[RandomForestConfig] = {

    val outputPayload = new ArrayBuffer[RandomForestConfig]()

    // figure out the number of permutations to generate

    /**
      * General Guidelines:
      *
      * - For String / Boolean values: Re-use as an iterator to continue to select index positions through the loop.
      * - For Continuous Variables:
      * -- mode: "Linear" - Generate uniformly sized Arrays, then build by index position to create the config.
      * -- mode: "Random" - Generate uniformly sized Array, build by random combination without replacement.
      */


    val impurityValues = _modelType match {
      case "regressor" => List("variance")
      case _ => stringBoundaries("impurity")
    }

    val impurityDistincts = impurityValues.length



    // Main builder loop
    for (i <- 1 to _permutationCount) {




    }



  }






}

case class NumericArrayCollection(
                                 selectedPayload: Array[Double],
                                 remainingPayload: Array[Array[Double]]
                                 )