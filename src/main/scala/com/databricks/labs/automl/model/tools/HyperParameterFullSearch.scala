package com.databricks.labs.automl.model.tools

import com.databricks.labs.automl.model.tools.structures._
import com.databricks.labs.automl.params._

import scala.collection.mutable.ArrayBuffer

class HyperParameterFullSearch extends Defaults with ModelConfigGenerators {

  var _modelFamily = ""
  var _modelType = ""
  var _permutationCount = 10
  var _indexMixingMode = "linear"
  var _arraySeed = 42L

  private val allowableMixingModes = List("linear", "random")

  def setModelFamily(value: String): this.type = {
    require(
      _supportedModels.contains(value),
      s"${this.getClass.toString} error! Model Family $value is not supported." +
        s"\n\t Supported families: ${_supportedModels.mkString(", ")}"
    )
    _modelFamily = value
    this
  }

  def setModelType(value: String): this.type = {
    value match {
      case "classifier" => _modelType = value
      case "regressor"  => _modelType = value
      case _ =>
        throw new UnsupportedOperationException(
          s"Model type $value is not supported."
        )
    }
    this
  }

  def setPermutationCount(value: Int): this.type = {
    _permutationCount = value
    this
  }

  def setIndexMixingMode(value: String): this.type = {
    require(
      allowableMixingModes.contains(value),
      s"Index Mixing mode $value is not supported.  Allowable modes are: " +
        s"${allowableMixingModes.mkString(", ")}"
    )
    _indexMixingMode = value
    this
  }

  def setArraySeed(value: Long): this.type = {
    _arraySeed = value
    this
  }

  def getModelFamily: String = _modelFamily
  def getModelType: String = _modelType
  def getPermutationCount: Int = _permutationCount
  def getIndexMixingMode: String = _indexMixingMode
  def getArraySeed: Long = _arraySeed

  /**
    * Method for generating a geometric space search for a first-generation hyper parameter generation for RandomForest
    * @param numericBoundaries The allowable restrictive space for the numeric hyper parameters
    * @param stringBoundaries The allowable values for string-based hyper parameters
    * @return An Array of Hyperparameter settings for RandomForest algorithms.
    */
  def initialGenerationSeedRandomForest(
    numericBoundaries: Map[String, (Double, Double)],
    stringBoundaries: Map[String, List[String]]
  ): Array[RandomForestConfig] = {

    var outputPayload = new ArrayBuffer[RandomForestConfig]()

    val impurityValues = _modelType match {
      case "regressor" => List("variance")
      case _           => stringBoundaries("impurity")
    }

    // Set the config object
    val rfConfig = PermutationConfiguration(
      modelType = _modelType,
      permutationTarget = _permutationCount,
      numericBoundaries = numericBoundaries,
      stringBoundaries = stringBoundaries
    )

    // Generate the permutation collections

    val generatedArrays = randomForestNumericArrayGenerator(rfConfig)

    // Create some index values
    var _impurityIdx = 0
    var _featureSubsetStrategyIdx = 0

    var numericArrays = Array(
      generatedArrays.numTreesArray,
      generatedArrays.maxBinsArray,
      generatedArrays.maxDepthArray,
      generatedArrays.minInfoGainArray,
      generatedArrays.subSamplingRateArray
    )

    // Main builder loop
    for (i <- 1 to _permutationCount) {

      val selectedIndeces = _indexMixingMode match {
        case "linear" => staticIndexSelection(numericArrays)
        case "random" => randomIndexSelection(numericArrays)
        case _ =>
          throw new UnsupportedOperationException(
            s"index mixing mode ${_indexMixingMode} is not supported."
          )
      }

      numericArrays = selectedIndeces.remainingPayload

      // Handle the string value selections
      val impurityLoop = selectStringIndex(impurityValues, _impurityIdx)

      _impurityIdx = impurityLoop.IndexCounterStatus

      val featureSubsetStrategyLoop = selectStringIndex(
        stringBoundaries("featureSubsetStrategy"),
        _featureSubsetStrategyIdx
      )

      _featureSubsetStrategyIdx = featureSubsetStrategyLoop.IndexCounterStatus

      outputPayload += RandomForestConfig(
        numTrees = selectedIndeces.selectedPayload(0).toInt,
        impurity = impurityLoop.selectedStringValue,
        maxBins = selectedIndeces.selectedPayload(1).toInt,
        maxDepth = selectedIndeces.selectedPayload(2).toInt,
        minInfoGain = selectedIndeces.selectedPayload(3),
        subSamplingRate = selectedIndeces.selectedPayload(4),
        featureSubsetStrategy = featureSubsetStrategyLoop.selectedStringValue
      )
      _impurityIdx += 1
      _featureSubsetStrategyIdx += 1
    }

    outputPayload.result.toArray

  }

  /**
    * Method for generating a geometric search space for a first-generation hyper parameter generation for LightGBM
    * @param numericBoundaries LightGBM numeric search space boundaries
    * @param stringBoundaries LightGBM string search space boundaries
    * @return An array of LightGBM configs
    * @since 0.6.1
    * @author Ben Wilson, Databricks
    * @throws UnsupportedOperationException if the index mixing mode supplied is invalid.
    */
  @throws(classOf[UnsupportedOperationException])
  def initialGenerationSeedLightGBM(
    numericBoundaries: Map[String, (Double, Double)],
    stringBoundaries: Map[String, List[String]]
  ): Array[LightGBMConfig] = {

    var outputPayload = new ArrayBuffer[LightGBMConfig]()

    val lightGBMConfig = PermutationConfiguration(
      modelType = _modelType,
      permutationTarget = _permutationCount,
      numericBoundaries = numericBoundaries,
      stringBoundaries = stringBoundaries
    )

    val generatedArrays = lightGBMNumericArrayGenerator(lightGBMConfig)

    var _boostFromAverageIdx = 0
    var _boostingTypeIdx = 0

    var numericArrays = Array(
      generatedArrays.baggingFractionArray,
      generatedArrays.baggingFreqArray,
      generatedArrays.featureFractionArray,
      generatedArrays.learningRateArray,
      generatedArrays.maxBinArray,
      generatedArrays.maxDepthArray,
      generatedArrays.minSumHessianInLeafArray,
      generatedArrays.numIterationsArray,
      generatedArrays.numLeavesArray,
      generatedArrays.lambdaL1Array,
      generatedArrays.lambdaL2Array,
      generatedArrays.alphaArray
    )

    for (i <- 1 to _permutationCount) {
      val selectedIndeces = _indexMixingMode match {
        case "linear" => staticIndexSelection(numericArrays)
        case "random" => randomIndexSelection(numericArrays)
        case _ =>
          throw new UnsupportedOperationException(
            s" Index mixing mode ${_indexMixingMode} is not supported."
          )
      }
      numericArrays = selectedIndeces.remainingPayload

      val boostFromAverageLoop = selectCoinFlip(_boostFromAverageIdx)
      val boostingTypeLoop =
        selectStringIndex(stringBoundaries("boostingType"), _boostingTypeIdx)
      _boostingTypeIdx = boostingTypeLoop.IndexCounterStatus

      outputPayload += LightGBMConfig(
        baggingFraction = selectedIndeces.selectedPayload(0),
        baggingFreq = selectedIndeces.selectedPayload(1).toInt,
        featureFraction = selectedIndeces.selectedPayload(2),
        learningRate = selectedIndeces.selectedPayload(3),
        maxBin = selectedIndeces.selectedPayload(4).toInt,
        maxDepth = selectedIndeces.selectedPayload(5).toInt,
        minSumHessianInLeaf = selectedIndeces.selectedPayload(6),
        numIterations = selectedIndeces.selectedPayload(7).toInt,
        numLeaves = selectedIndeces.selectedPayload(8).toInt,
        boostFromAverage = boostFromAverageLoop,
        lambdaL1 = selectedIndeces.selectedPayload(9),
        lambdaL2 = selectedIndeces.selectedPayload(10),
        alpha = selectedIndeces.selectedPayload(11),
        boostingType = boostingTypeLoop.selectedStringValue
      )
      _boostFromAverageIdx += 1
      _boostingTypeIdx += 1

    }

    outputPayload.result.toArray

  }

  /**
    * Method for generating a geometric search space for a first-generation hyper parameter generation for DecisionTrees
    * @param numericBoundaries numeric bounds restrictions
    * @param stringBoundaries string value restrictions
    * @return An Array of Hyperparameter settings for DecisionTrees algorithms
    */
  def initialGenerationSeedTrees(
    numericBoundaries: Map[String, (Double, Double)],
    stringBoundaries: Map[String, List[String]]
  ): Array[TreesConfig] = {

    var outputPayload = new ArrayBuffer[TreesConfig]()

    val impurityValues = _modelType match {
      case "regressor" => List("variance")
      case _           => stringBoundaries("impurity")
    }

    val treesConfig = PermutationConfiguration(
      modelType = _modelType,
      permutationTarget = _permutationCount,
      numericBoundaries = numericBoundaries,
      stringBoundaries = stringBoundaries
    )

    val generatedArrays = treesNumericArrayGenerator(treesConfig)

    var _impurityIdx = 0

    var numericArrays = Array(
      generatedArrays.maxBinsArray,
      generatedArrays.maxDepthArray,
      generatedArrays.minInfoGainArray,
      generatedArrays.minInstancesPerNodeArray
    )

    for (i <- 1 to _permutationCount) {
      val selectedIndeces = _indexMixingMode match {
        case "linear" => staticIndexSelection(numericArrays)
        case "random" => randomIndexSelection(numericArrays)
        case _ =>
          throw new UnsupportedOperationException(
            s"Index mixing mode ${_indexMixingMode} is not supported."
          )
      }

      numericArrays = selectedIndeces.remainingPayload

      val impurityLoop = selectStringIndex(impurityValues, _impurityIdx)
      _impurityIdx = impurityLoop.IndexCounterStatus

      outputPayload += TreesConfig(
        impurity = impurityLoop.selectedStringValue,
        maxBins = selectedIndeces.selectedPayload(0).toInt,
        maxDepth = selectedIndeces.selectedPayload(1).toInt,
        minInfoGain = selectedIndeces.selectedPayload(2),
        minInstancesPerNode = selectedIndeces.selectedPayload(3).toInt
      )
      _impurityIdx += 1
    }

    outputPayload.result.toArray

  }

  def initialGenerationSeedGBT(
    numericBoundaries: Map[String, (Double, Double)],
    stringBoundaries: Map[String, List[String]]
  ): Array[GBTConfig] = {
    var outputPayload = new ArrayBuffer[GBTConfig]()

    val impurityValues = _modelType match {
      case "regressor" => List("variance")
      case _           => stringBoundaries("impurity")
    }
    val lossTypeValues = _modelType match {
      case "regressor" => List("squared", "absolute")
      case _           => stringBoundaries("lossType")
    }

    val gbtConfig = PermutationConfiguration(
      modelType = _modelType,
      permutationTarget = _permutationCount,
      numericBoundaries = numericBoundaries,
      stringBoundaries = stringBoundaries
    )

    val generatedArrays = gbtNumericArrayGenerator(gbtConfig)

    var _impurityIdx = 0
    var _lossTypeIdx = 0

    var numericArrays = Array(
      generatedArrays.maxBinsArray,
      generatedArrays.maxDepthArray,
      generatedArrays.maxIterArray,
      generatedArrays.minInfoGainArray,
      generatedArrays.minInstancesPerNodeArray,
      generatedArrays.stepSizeArray
    )

    for (i <- 1 to _permutationCount) {
      val selectedIndeces = _indexMixingMode match {
        case "linear" => staticIndexSelection(numericArrays)
        case "random" => randomIndexSelection(numericArrays)
        case _ =>
          throw new UnsupportedOperationException(
            s"Index mixing mode ${_indexMixingMode} is not supported."
          )
      }

      numericArrays = selectedIndeces.remainingPayload

      val impurityLoop = selectStringIndex(impurityValues, _impurityIdx)
      val lossTypeLoop = selectStringIndex(lossTypeValues, _lossTypeIdx)
      _impurityIdx = impurityLoop.IndexCounterStatus
      _lossTypeIdx = lossTypeLoop.IndexCounterStatus

      outputPayload += GBTConfig(
        impurity = impurityLoop.selectedStringValue,
        lossType = lossTypeLoop.selectedStringValue,
        maxBins = selectedIndeces.selectedPayload(0).toInt,
        maxDepth = selectedIndeces.selectedPayload(1).toInt,
        maxIter = selectedIndeces.selectedPayload(2).toInt,
        minInfoGain = selectedIndeces.selectedPayload(3),
        minInstancesPerNode = selectedIndeces.selectedPayload(4).toInt,
        stepSize = selectedIndeces.selectedPayload(5)
      )
      _impurityIdx += 1
      _lossTypeIdx += 1
    }
    outputPayload.result.toArray

  }

  def initialGenerationSeedLinearRegression(
    numericBoundaries: Map[String, (Double, Double)],
    stringBoundaries: Map[String, List[String]]
  ): Array[LinearRegressionConfig] = {
    var outputPayload = new ArrayBuffer[LinearRegressionConfig]()

    val linearRegressionConfig = PermutationConfiguration(
      modelType = _modelType,
      permutationTarget = _permutationCount,
      numericBoundaries = numericBoundaries,
      stringBoundaries = stringBoundaries
    )

    val generatedArrays = linearRegressionNumericArrayGenerator(
      linearRegressionConfig
    )

    var _fitInterceptIdx = 0
    var _standardizationIdx = 0
    var _lossIdx = 0

    var numericArrays = Array(
      generatedArrays.elasticNetParamsArray,
      generatedArrays.maxIterArray,
      generatedArrays.regParamArray,
      generatedArrays.toleranceArray
    )

    for (i <- 1 to _permutationCount) {
      val selectedIndeces = _indexMixingMode match {
        case "linear" => staticIndexSelection(numericArrays)
        case "random" => randomIndexSelection(numericArrays)
        case _ =>
          throw new UnsupportedOperationException(
            s"Index mixing mode ${_indexMixingMode} is not supported."
          )
      }

      numericArrays = selectedIndeces.remainingPayload

      val fitInterceptLoop = selectCoinFlip(_fitInterceptIdx)
      val standardizationLoop = selectCoinFlip(_standardizationIdx)
      val lossLoop = selectStringIndex(stringBoundaries("loss"), _lossIdx)
      _lossIdx = lossLoop.IndexCounterStatus

      /**
        * For Linear Regression, the loss setting of 'huber' does not permit regularization of elasticnet or L1.
        * It must be set to L2 regularization (elasticNetParams == 0.0) to function.
        */
      val loss = lossLoop.selectedStringValue
      val elasticNetParams = loss match {
        case "huber" => 0.0
        case _       => selectedIndeces.selectedPayload(0)
      }

      outputPayload += LinearRegressionConfig(
        loss = loss,
        elasticNetParams = elasticNetParams,
        fitIntercept = fitInterceptLoop,
        maxIter = selectedIndeces.selectedPayload(1).toInt,
        regParam = selectedIndeces.selectedPayload(2),
        standardization = standardizationLoop,
        tolerance = selectedIndeces.selectedPayload(3)
      )
      _lossIdx += 1
      _standardizationIdx += 1
      _fitInterceptIdx += 1
    }
    outputPayload.result.toArray
  }

  def initialGenerationSeedLogisticRegression(
    numericBoundaries: Map[String, (Double, Double)]
  ): Array[LogisticRegressionConfig] = {

    var outputPayload = new ArrayBuffer[LogisticRegressionConfig]()

    val logisticRegressionConfig = PermutationConfiguration(
      modelType = _modelType,
      permutationTarget = _permutationCount,
      numericBoundaries = numericBoundaries,
      stringBoundaries = Map[String, List[String]]()
    )

    val generatedArrays = logisticRegressionNumericArrayGenerator(
      logisticRegressionConfig
    )

    var _fitInterceptIdx = 0
    var _standardizationIdx = 0

    var numericArrays = Array(
      generatedArrays.elasticNetParamsArray,
      generatedArrays.maxIterArray,
      generatedArrays.regParamArray,
      generatedArrays.toleranceArray
    )

    for (i <- 1 to _permutationCount) {
      val selectedIndeces = _indexMixingMode match {
        case "linear" => staticIndexSelection(numericArrays)
        case "random" => randomIndexSelection(numericArrays)
        case _ =>
          throw new UnsupportedOperationException(
            s"Index mixing mode ${_indexMixingMode} is not supported."
          )
      }

      numericArrays = selectedIndeces.remainingPayload

      val fitInterceptLoop = selectCoinFlip(_fitInterceptIdx)
      val standardizationLoop = selectCoinFlip(_standardizationIdx)

      outputPayload += LogisticRegressionConfig(
        elasticNetParams = selectedIndeces.selectedPayload(0),
        fitIntercept = fitInterceptLoop,
        maxIter = selectedIndeces.selectedPayload(1).toInt,
        regParam = selectedIndeces.selectedPayload(2),
        standardization = standardizationLoop,
        tolerance = selectedIndeces.selectedPayload(3)
      )
      _standardizationIdx += 1
      _fitInterceptIdx += 1
    }
    outputPayload.result.toArray

  }

  def initialGenerationSeedSVM(
    numericBoundaries: Map[String, (Double, Double)]
  ): Array[SVMConfig] = {

    var outputPayload = new ArrayBuffer[SVMConfig]()

    val svmConfig = PermutationConfiguration(
      modelType = _modelType,
      permutationTarget = _permutationCount,
      numericBoundaries = numericBoundaries,
      stringBoundaries = Map[String, List[String]]()
    )

    val generatedArrays = svmNumericArrayGenerator(svmConfig)

    var _fitInterceptIdx = 0
    var _standardizationIdx = 0

    var numericArrays = Array(
      generatedArrays.maxIterArray,
      generatedArrays.regParamArray,
      generatedArrays.toleranceArray
    )

    for (i <- 1 to _permutationCount) {
      val selectedIndeces = _indexMixingMode match {
        case "linear" => staticIndexSelection(numericArrays)
        case "random" => randomIndexSelection(numericArrays)
        case _ =>
          throw new UnsupportedOperationException(
            s"Index mixing mode ${_indexMixingMode} is not supported."
          )
      }

      numericArrays = selectedIndeces.remainingPayload

      val fitInterceptLoop = selectCoinFlip(_fitInterceptIdx)
      val standardizationLoop = selectCoinFlip(_standardizationIdx)

      outputPayload += SVMConfig(
        fitIntercept = fitInterceptLoop,
        maxIter = selectedIndeces.selectedPayload(0).toInt,
        regParam = selectedIndeces.selectedPayload(1),
        standardization = standardizationLoop,
        tolerance = selectedIndeces.selectedPayload(2)
      )
      _standardizationIdx += 1
      _fitInterceptIdx += 1
    }
    outputPayload.result.toArray

  }

  def initialGenerationSeedXGBoost(
    numericBoundaries: Map[String, (Double, Double)]
  ): Array[XGBoostConfig] = {

    var outputPayload = new ArrayBuffer[XGBoostConfig]()

    val xgboostConfig = PermutationConfiguration(
      modelType = _modelType,
      permutationTarget = _permutationCount,
      numericBoundaries = numericBoundaries,
      stringBoundaries = Map[String, List[String]]()
    )

    val generatedArrays = xgboostNumericArrayGenerator(xgboostConfig)

    var numericArrays = Array(
      generatedArrays.alphaArray,
      generatedArrays.etaArray,
      generatedArrays.gammaArray,
      generatedArrays.lambdaArray,
      generatedArrays.maxDepthArray,
      generatedArrays.subSampleArray,
      generatedArrays.minChildWeightArray,
      generatedArrays.numRoundArray,
      generatedArrays.maxBinsArray,
      generatedArrays.trainTestRatioArray
    )

    for (i <- 1 to _permutationCount) {
      val selectedIndeces = _indexMixingMode match {
        case "linear" => staticIndexSelection(numericArrays)
        case "random" => randomIndexSelection(numericArrays)
        case _ =>
          throw new UnsupportedOperationException(
            s"Index mixing mode ${_indexMixingMode} is not supported."
          )
      }

      numericArrays = selectedIndeces.remainingPayload

      outputPayload += XGBoostConfig(
        alpha = selectedIndeces.selectedPayload(0),
        eta = selectedIndeces.selectedPayload(1),
        gamma = selectedIndeces.selectedPayload(2),
        lambda = selectedIndeces.selectedPayload(3),
        maxDepth = selectedIndeces.selectedPayload(4).toInt,
        subSample = selectedIndeces.selectedPayload(5),
        minChildWeight = selectedIndeces.selectedPayload(6),
        numRound = selectedIndeces.selectedPayload(7).toInt,
        maxBins = selectedIndeces.selectedPayload(8).toInt,
        trainTestRatio = selectedIndeces.selectedPayload(9)
      )

    }
    outputPayload.result.toArray

  }

  def initialGenerationSeedMLPC(
    numericBoundaries: Map[String, (Double, Double)],
    stringBoundaries: Map[String, List[String]],
    inputFeatureSize: Int,
    distinctClasses: Int
  ): Array[MLPCConfig] = {

    var outputPayload = new ArrayBuffer[MLPCConfig]()

    val mlpcConfig = MLPCPermutationConfiguration(
      permutationTarget = _permutationCount,
      numericBoundaries = numericBoundaries,
      stringBoundaries = stringBoundaries,
      inputFeatureSize = inputFeatureSize,
      distinctClasses = distinctClasses
    )

    var generatedArrays = mlpcNumericArrayGenerator(mlpcConfig)

    var _solverIdx = 0

    for (i <- 1 to _permutationCount) {
      val selectedIndeces = _indexMixingMode match {
        case "linear" => mlpcStaticIndexSelection(generatedArrays)
        case "random" => mlpcRandomIndexSelection(generatedArrays)
        case _ =>
          throw new UnsupportedOperationException(
            s"Index mixing mode ${_indexMixingMode} is not supported."
          )
      }

      generatedArrays = selectedIndeces.remainingPayloads

      val solverLoop = selectStringIndex(stringBoundaries("solver"), _solverIdx)
      _solverIdx = solverLoop.IndexCounterStatus

      outputPayload += MLPCConfig(
        layers = selectedIndeces.selectedPayload.layers,
        maxIter = selectedIndeces.selectedPayload.maxIter,
        solver = solverLoop.selectedStringValue,
        stepSize = selectedIndeces.selectedPayload.stepSize,
        tolerance = selectedIndeces.selectedPayload.tolerance
      )
      _solverIdx += 1
    }
    outputPayload.result.toArray
  }

}
