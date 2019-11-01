package com.databricks.labs.automl.model.tools

import com.databricks.labs.automl.model.tools.structures._
import com.databricks.labs.automl.params._
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer

class PostModelingOptimization
    extends Defaults
    with ModelConfigGenerators
    with SparkSessionWrapper {

  private final val PERMUTATION_FACTOR: Int = 10
  private final val PREDICTION_COL: String = "prediction"
  private final val supportedOptimizationStrategies: List[String] =
    List("minimize", "maximize")

  var _modelFamily = ""
  var _modelType = ""
  var _hyperParameterSpaceCount = 100000
  var _numericBoundaries: Map[String, (Double, Double)] = _
  var _stringBoundaries: Map[String, List[String]] = _
  var _seed: Long = 42L
  var _optimizationStrategy: String = "maximize"

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

  def setHyperParameterSpaceCount(value: Int): this.type = {
    if (value > 500000)
      println(
        "WARNING! Setting permutation counts above 500,000 will put stress on the driver."
      )
    if (value > 1000000)
      throw new UnsupportedOperationException(
        s"Setting permutation above 1,000,000 is not supported" +
          s" due to runtime considerations.  $value is too large of a value."
      )
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

  def setOptimizationStrategy(value: String): this.type = {

    require(
      supportedOptimizationStrategies.contains(value),
      s"Optimization Strategy for Post Modeling Optimization " +
        s"$value is not supported.  Must be one of: ${supportedOptimizationStrategies.mkString(", ")}."
    )
    _optimizationStrategy = value
    this
  }

  def getModelFamily: String = _modelFamily

  def getModelType: String = _modelType

  def getHyperParameterSpaceCount: Int = _hyperParameterSpaceCount

  def getNumericBoundaries: Map[String, (Double, Double)] = _numericBoundaries

  def getStringBoundaries: Map[String, List[String]] = _stringBoundaries

  def getSeed: Long = _seed

  def getOptimizationStrategy: String = _optimizationStrategy

  private def generateGenericSearchSpace(): PermutationConfiguration = {
    val calculatedPermutationValue = getPermutationCounts(
      _hyperParameterSpaceCount,
      _numericBoundaries.size
    ) +
      stringBoundaryPermutationCalculator(_stringBoundaries)

    PermutationConfiguration(
      modelType = _modelType,
      permutationTarget = calculatedPermutationValue,
      numericBoundaries = _numericBoundaries,
      stringBoundaries = _stringBoundaries
    )
  }

  private def euclideanRestrict(df: DataFrame,
                                topPredictions: Int,
                                additionalFields: Array[String] =
                                  Array[String]()): DataFrame = {

    EuclideanSpaceSearch(
      df,
      _numericBoundaries.keys.toArray,
      _stringBoundaries.keys.toArray,
      topPredictions,
      additionalFields
    )

  }

  /**
    * Private method for returning the top n hyper parameters based on the direction of optimization that should occur
    * for the metric being evaluated.
    * @param pipeline ML Pipeline object
    * @param data DataFrame continaing the hyper parameters to predict performance for
    * @param topPredictions The number of potential candidates to return.
    * @return DataFrame of relevant candidates
    * @since 0.6.1
    * @author Ben Wilson, Databricks
    */
  private def transformAndLimit(pipeline: PipelineModel,
                                data: DataFrame,
                                topPredictions: Int): DataFrame = {

    _optimizationStrategy match {
      case "minimize" =>
        pipeline
          .transform(data)
          .orderBy(col(PREDICTION_COL).asc)
          .limit(topPredictions * PERMUTATION_FACTOR)
      case _ =>
        pipeline
          .transform(data)
          .orderBy(col(PREDICTION_COL).desc)
          .limit(topPredictions * PERMUTATION_FACTOR)
    }

  }

  //RANDOM FOREST METHODS
  /**
    * Generates an array of RandomForestConfig hyper parameters to meet the configured target size
    * @return a distinct array of RandomForestConfig's
    */
  protected[tools] def generateRandomForestSearchSpace()
    : Array[RandomForestConfig] = {
    // Generate the Permutations
    val permutationsArray = randomForestPermutationGenerator(
      generateGenericSearchSpace(),
      _hyperParameterSpaceCount,
      _seed
    )

    permutationsArray.distinct
  }

  def generateRandomForestSearchSpaceAsDataFrame(): DataFrame = {

    spark.createDataFrame(generateRandomForestSearchSpace())

  }

  protected[tools] def randomForestResultMapping(
    results: Array[GenericModelReturn]
  ): DataFrame = {

    val builder = new ArrayBuffer[RandomForestModelRunReport]()

    results.foreach { x =>
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

  def randomForestPrediction(modelingResults: Array[GenericModelReturn],
                             modelType: String,
                             topPredictions: Int): Array[RandomForestConfig] = {

    val inferenceDataSet = randomForestResultMapping(modelingResults)

    val fittedPipeline = new PostModelingPipelineBuilder(inferenceDataSet)
      .setModelType(modelType)
      .setNumericBoundaries(_numericBoundaries)
      .setStringBoundaries(_stringBoundaries)
      .regressionModelForPermutationTest()

    val fullSearchSpaceDataSet = generateRandomForestSearchSpaceAsDataFrame()

    val restrictedData =
      transformAndLimit(fittedPipeline, fullSearchSpaceDataSet, topPredictions)

    convertRandomForestResultToConfig(
      euclideanRestrict(restrictedData, topPredictions)
    )

  }

  //DECISION TREE METHODS

  protected[tools] def generateTreesSearchSpace(): Array[TreesConfig] = {

    val permutationsArray = treesPermutationGenerator(
      generateGenericSearchSpace(),
      _hyperParameterSpaceCount,
      _seed
    )
    permutationsArray.distinct
  }

  protected[tools] def generateTreesSearchSpaceAsDataFrame(): DataFrame = {
    spark.createDataFrame(generateTreesSearchSpace())
  }

  protected[tools] def treesResultMapping(
    results: Array[GenericModelReturn]
  ): DataFrame = {

    val builder = new ArrayBuffer[TreesModelRunReport]()

    results.foreach { x =>
      val hyperParams = x.hyperParams
      builder += TreesModelRunReport(
        impurity = hyperParams("impurity").toString,
        maxBins = hyperParams("maxBins").toString.toInt,
        maxDepth = hyperParams("maxDepth").toString.toInt,
        minInfoGain = hyperParams("minInfoGain").toString.toDouble,
        minInstancesPerNode =
          hyperParams("minInstancesPerNode").toString.toDouble,
        score = x.score
      )
    }
    spark.createDataFrame(builder.result.toArray)
  }

  def treesPrediction(modelingResults: Array[GenericModelReturn],
                      modelType: String,
                      topPredictions: Int): Array[TreesConfig] = {
    val inferenceDataSet = treesResultMapping(modelingResults)

    val fittedPipeline = new PostModelingPipelineBuilder(inferenceDataSet)
      .setModelType(modelType)
      .setNumericBoundaries(_numericBoundaries)
      .setStringBoundaries(_stringBoundaries)
      .regressionModelForPermutationTest()

    val fullSearchSpaceDataSet = generateTreesSearchSpaceAsDataFrame()

    val restrictedData =
      transformAndLimit(fittedPipeline, fullSearchSpaceDataSet, topPredictions)

    convertTreesResultToConfig(
      euclideanRestrict(restrictedData, topPredictions)
    )
  }

  //GBT METHODS

  protected[tools] def generateGBTSearchSpace(): Array[GBTConfig] = {

    val permutationsArray = gbtPermutationGenerator(
      generateGenericSearchSpace(),
      _hyperParameterSpaceCount,
      _seed
    )
    permutationsArray.distinct
  }

  protected[tools] def generateGBTSearchSpaceAsDataFrame(): DataFrame = {
    spark.createDataFrame(generateGBTSearchSpace())
  }

  protected[tools] def gbtResultMapping(
    results: Array[GenericModelReturn]
  ): DataFrame = {

    val builder = new ArrayBuffer[GBTModelRunReport]()

    results.foreach { x =>
      val hyperParams = x.hyperParams
      builder += GBTModelRunReport(
        impurity = hyperParams("impurity").toString,
        lossType = hyperParams("lossType").toString,
        maxBins = hyperParams("maxBins").toString.toInt,
        maxDepth = hyperParams("maxDepth").toString.toInt,
        maxIter = hyperParams("maxIter").toString.toInt,
        minInfoGain = hyperParams("minInfoGain").toString.toDouble,
        minInstancesPerNode = hyperParams("minInstancesPerNode").toString.toInt,
        stepSize = hyperParams("stepSize").toString.toDouble,
        score = x.score
      )
    }
    spark.createDataFrame(builder.result.toArray)
  }

  def gbtPrediction(modelingResults: Array[GenericModelReturn],
                    modelType: String,
                    topPredictions: Int): Array[GBTConfig] = {
    val inferenceDataSet = gbtResultMapping(modelingResults)

    val fittedPipeline = new PostModelingPipelineBuilder(inferenceDataSet)
      .setModelType(modelType)
      .setNumericBoundaries(_numericBoundaries)
      .setStringBoundaries(_stringBoundaries)
      .regressionModelForPermutationTest()

    val fullSearchSpaceDataSet = generateGBTSearchSpaceAsDataFrame()

    val restrictedData =
      transformAndLimit(fittedPipeline, fullSearchSpaceDataSet, topPredictions)

    convertGBTResultToConfig(euclideanRestrict(restrictedData, topPredictions))
  }

  //LINEAR REGRESSION METHODS

  protected[tools] def generateLinearRegressionSearchSpace()
    : Array[LinearRegressionConfig] = {

    val permutationsArray = linearRegressionPermutationGenerator(
      generateGenericSearchSpace(),
      _hyperParameterSpaceCount,
      _seed
    )
    permutationsArray.distinct
  }

  protected[tools] def generateLinearRegressionSearchSpaceAsDataFrame()
    : DataFrame = {
    spark.createDataFrame(generateLinearRegressionSearchSpace())
  }

  protected[tools] def linearRegressionResultMapping(
    results: Array[GenericModelReturn]
  ): DataFrame = {

    val builder = new ArrayBuffer[LinearRegressionModelRunReport]()

    results.foreach { x =>
      val hyperParams = x.hyperParams
      builder += LinearRegressionModelRunReport(
        elasticNetParams = hyperParams("elasticNetParams").toString.toDouble,
        fitIntercept = hyperParams("fitIntercept").toString.toBoolean,
        loss = hyperParams("loss").toString,
        maxIter = hyperParams("maxIter").toString.toInt,
        regParam = hyperParams("regParam").toString.toDouble,
        standardization = hyperParams("standardization").toString.toBoolean,
        tolerance = hyperParams("tolerance").toString.toDouble,
        score = x.score
      )
    }
    spark.createDataFrame(builder.result.toArray)
  }

  def linearRegressionPrediction(
    modelingResults: Array[GenericModelReturn],
    modelType: String,
    topPredictions: Int
  ): Array[LinearRegressionConfig] = {
    val inferenceDataSet = linearRegressionResultMapping(modelingResults)

    val fittedPipeline = new PostModelingPipelineBuilder(inferenceDataSet)
      .setModelType(modelType)
      .setNumericBoundaries(_numericBoundaries)
      .setStringBoundaries(_stringBoundaries)
      .regressionModelForPermutationTest()

    val fullSearchSpaceDataSet =
      generateLinearRegressionSearchSpaceAsDataFrame()

    val restrictedData =
      transformAndLimit(fittedPipeline, fullSearchSpaceDataSet, topPredictions)

    convertLinearRegressionResultToConfig(
      euclideanRestrict(
        restrictedData,
        topPredictions,
        Array("fitIntercept", "standardization")
      )
    )
  }

  //LOGISTIC REGRESSION METHODS

  protected[tools] def generateLogisticRegressionSearchSpace()
    : Array[LogisticRegressionConfig] = {

    val permutationsArray = logisticRegressionPermutationGenerator(
      generateGenericSearchSpace(),
      _hyperParameterSpaceCount,
      _seed
    )
    permutationsArray.distinct
  }

  protected[tools] def generateLogisticRegressionSearchSpaceAsDataFrame()
    : DataFrame = {
    spark.createDataFrame(generateLogisticRegressionSearchSpace())
  }

  protected[tools] def logisticRegressionResultMapping(
    results: Array[GenericModelReturn]
  ): DataFrame = {

    val builder = new ArrayBuffer[LogisticRegressionModelRunReport]()

    results.foreach { x =>
      val hyperParams = x.hyperParams
      builder += LogisticRegressionModelRunReport(
        elasticNetParams = hyperParams("elasticNetParams").toString.toDouble,
        fitIntercept = hyperParams("fitIntercept").toString.toBoolean,
        maxIter = hyperParams("maxIter").toString.toInt,
        regParam = hyperParams("regParam").toString.toDouble,
        standardization = hyperParams("standardization").toString.toBoolean,
        tolerance = hyperParams("tolerance").toString.toDouble,
        score = x.score
      )
    }
    spark.createDataFrame(builder.result.toArray)
  }

  def logisticRegressionPrediction(
    modelingResults: Array[GenericModelReturn],
    modelType: String,
    topPredictions: Int
  ): Array[LogisticRegressionConfig] = {
    val inferenceDataSet = logisticRegressionResultMapping(modelingResults)

    val fittedPipeline = new PostModelingPipelineBuilder(inferenceDataSet)
      .setModelType(modelType)
      .setNumericBoundaries(_numericBoundaries)
      .setStringBoundaries(_stringBoundaries)
      .regressionModelForPermutationTest()

    val fullSearchSpaceDataSet =
      generateLogisticRegressionSearchSpaceAsDataFrame()

    val restrictedData =
      transformAndLimit(fittedPipeline, fullSearchSpaceDataSet, topPredictions)

    convertLogisticRegressionResultToConfig(
      euclideanRestrict(
        restrictedData,
        topPredictions,
        Array("fitIntercept", "standardization")
      )
    )
  }

  //SUPPORT VECTOR MACHINES METHODS

  protected[tools] def generateSVMSearchSpace(): Array[SVMConfig] = {

    val permutationsArray = svmPermutationGenerator(
      generateGenericSearchSpace(),
      _hyperParameterSpaceCount,
      _seed
    )
    permutationsArray.distinct
  }

  protected[tools] def generateSVMSearchSpaceAsDataFrame(): DataFrame = {
    spark.createDataFrame(generateSVMSearchSpace())
  }

  protected[tools] def svmResultMapping(
    results: Array[GenericModelReturn]
  ): DataFrame = {

    val builder = new ArrayBuffer[SVMModelRunReport]()

    results.foreach { x =>
      val hyperParams = x.hyperParams
      builder += SVMModelRunReport(
        fitIntercept = hyperParams("fitIntercept").toString.toBoolean,
        maxIter = hyperParams("maxIter").toString.toInt,
        regParam = hyperParams("regParam").toString.toDouble,
        standardization = hyperParams("standardization").toString.toBoolean,
        tolerance = hyperParams("tolerance").toString.toDouble,
        score = x.score
      )
    }
    spark.createDataFrame(builder.result.toArray)
  }

  def svmPrediction(modelingResults: Array[GenericModelReturn],
                    modelType: String,
                    topPredictions: Int): Array[SVMConfig] = {
    val inferenceDataSet = svmResultMapping(modelingResults)

    val fittedPipeline = new PostModelingPipelineBuilder(inferenceDataSet)
      .setModelType(modelType)
      .setNumericBoundaries(_numericBoundaries)
      .setStringBoundaries(_stringBoundaries)
      .regressionModelForPermutationTest()

    val fullSearchSpaceDataSet = generateSVMSearchSpaceAsDataFrame()

    val restrictedData =
      transformAndLimit(fittedPipeline, fullSearchSpaceDataSet, topPredictions)

    convertSVMResultToConfig(
      euclideanRestrict(
        restrictedData,
        topPredictions,
        Array("fitIntercept", "standardization")
      )
    )
  }

  //XGBOOST METHODS

  protected[tools] def generateXGBoostSearchSpace(): Array[XGBoostConfig] = {

    val permutationsArray = xgboostPermutationGenerator(
      generateGenericSearchSpace(),
      _hyperParameterSpaceCount,
      _seed
    )
    permutationsArray.distinct
  }

  protected[tools] def generateXGBoostSearchSpaceAsDataFrame(): DataFrame = {
    spark.createDataFrame(generateXGBoostSearchSpace())
  }

  protected[tools] def xgBoostResultMapping(
    results: Array[GenericModelReturn]
  ): DataFrame = {

    val builder = new ArrayBuffer[XGBoostModelRunReport]()

    results.foreach { x =>
      val hyperParams = x.hyperParams
      builder += XGBoostModelRunReport(
        alpha = hyperParams("alpha").toString.toDouble,
        eta = hyperParams("eta").toString.toDouble,
        gamma = hyperParams("gamma").toString.toDouble,
        lambda = hyperParams("lambda").toString.toDouble,
        maxDepth = hyperParams("maxDepth").toString.toInt,
        subSample = hyperParams("subSample").toString.toDouble,
        minChildWeight = hyperParams("minChildWeight").toString.toDouble,
        numRound = hyperParams("numRound").toString.toInt,
        maxBins = hyperParams("maxBins").toString.toInt,
        trainTestRatio = hyperParams("trainTestRatio").toString.toDouble,
        score = x.score
      )
    }
    spark.createDataFrame(builder.result.toArray)
  }

  def xgBoostPrediction(modelingResults: Array[GenericModelReturn],
                        modelType: String,
                        topPredictions: Int): Array[XGBoostConfig] = {
    val inferenceDataSet = xgBoostResultMapping(modelingResults)

    val fittedPipeline = new PostModelingPipelineBuilder(inferenceDataSet)
      .setModelType(modelType)
      .setNumericBoundaries(_numericBoundaries)
      .setStringBoundaries(_stringBoundaries)
      .regressionModelForPermutationTest()

    val fullSearchSpaceDataSet = generateXGBoostSearchSpaceAsDataFrame()

    val restrictedData =
      transformAndLimit(fittedPipeline, fullSearchSpaceDataSet, topPredictions)

    convertXGBoostResultToConfig(
      euclideanRestrict(restrictedData, topPredictions)
    )
  }

  //LIGHTGBM METHODS

  protected[tools] def generateLightGBMSearchSpace(): Array[LightGBMConfig] = {

    val permutationsArray = lightGBMPermutationGenerator(
      generateGenericSearchSpace(),
      _hyperParameterSpaceCount,
      _seed
    )
    permutationsArray.distinct
  }

  protected[tools] def generateLightGBMSearchSpaceAsDataFrame(): DataFrame = {
    spark.createDataFrame(generateLightGBMSearchSpace())
  }

  protected[tools] def lightGBMResultMapping(
    results: Array[GenericModelReturn]
  ): DataFrame = {

    val builder = results.map { x =>
      val hyperParams = x.hyperParams
      LightGBMModelRunReport(
        baggingFraction = hyperParams("baggingFraction").toString.toDouble,
        baggingFreq = hyperParams("baggingFreq").toString.toInt,
        featureFraction = hyperParams("featureFraction").toString.toDouble,
        learningRate = hyperParams("learningRate").toString.toDouble,
        maxBin = hyperParams("maxBin").toString.toInt,
        maxDepth = hyperParams("maxDepth").toString.toInt,
        minSumHessianInLeaf =
          hyperParams("minSumHessianInLeaf").toString.toDouble,
        numIterations = hyperParams("numIterations").toString.toInt,
        numLeaves = hyperParams("numLeaves").toString.toInt,
        boostFromAverage = hyperParams("boostFromAverage").toString.toBoolean,
        lambdaL1 = hyperParams("lambdaL1").toString.toDouble,
        lambdaL2 = hyperParams("lambdaL2").toString.toDouble,
        alpha = hyperParams("alpha").toString.toDouble,
        boostingType = hyperParams("boostingType").toString,
        score = x.score
      )
    }
    spark.createDataFrame(builder)
  }

  def lightGBMPrediction(modelingResults: Array[GenericModelReturn],
                         modelType: String,
                         topPredictions: Int): Array[LightGBMConfig] = {

    val inferenceDataSet = lightGBMResultMapping(modelingResults)

    val fittedPipeline = new PostModelingPipelineBuilder(inferenceDataSet)
      .setModelType(modelType)
      .setNumericBoundaries(_numericBoundaries)
      .setStringBoundaries(_stringBoundaries)
      .regressionModelForPermutationTest()

    val fullSearchSpaceDataSet = generateLightGBMSearchSpaceAsDataFrame()

    val restrictedData =
      transformAndLimit(fittedPipeline, fullSearchSpaceDataSet, topPredictions)

    convertLightGBMResultToConfig(
      euclideanRestrict(restrictedData, topPredictions)
    )

  }

  //MLPC METHODS

  protected[tools] def generateMLPCSearchSpace(
    inputFeatureSize: Int,
    classCount: Int
  ): Array[MLPCModelingConfig] = {

    val mlpcSearchSpace = MLPCPermutationConfiguration(
      permutationTarget = getPermutationCounts(
        _hyperParameterSpaceCount,
        _numericBoundaries.size
      ) +
        stringBoundaryPermutationCalculator(_stringBoundaries),
      numericBoundaries = _numericBoundaries,
      stringBoundaries = _stringBoundaries,
      inputFeatureSize = inputFeatureSize,
      distinctClasses = classCount
    )

    val permutationsArray = mlpcPermutationGenerator(
      mlpcSearchSpace,
      _hyperParameterSpaceCount,
      _seed
    )
    permutationsArray.distinct
  }

  protected[tools] def generateMLPCSearchSpaceAsDataFrame(
    inputFeatureSize: Int,
    classCount: Int
  ): DataFrame = {
    spark.createDataFrame(generateMLPCSearchSpace(inputFeatureSize, classCount))
  }

  protected[tools] def mlpcResultMapping(
    results: Array[GenericModelReturn]
  ): DataFrame = {

    val builder = new ArrayBuffer[MLPCModelRunReport]()

    results.foreach { x =>
      val hyperParams = x.hyperParams
      val (layerCount, hiddenLayerSizeAdjust) =
        mlpcLayersExtractor(hyperParams("layers").asInstanceOf[Array[Int]])
      builder += MLPCModelRunReport(
        layers = layerCount,
        maxIter = hyperParams("maxIter").toString.toInt,
        solver = hyperParams("solver").toString,
        stepSize = hyperParams("stepSize").toString.toDouble,
        tolerance = hyperParams("tolerance").toString.toDouble,
        hiddenLayerSizeAdjust = hiddenLayerSizeAdjust,
        score = x.score
      )
    }
    spark.createDataFrame(builder.result.toArray)
  }

  def mlpcPrediction(modelingResults: Array[GenericModelReturn],
                     modelType: String,
                     topPredictions: Int,
                     featureInputSize: Int,
                     classDistinctCount: Int): Array[MLPCConfig] = {

    val inferenceDataSet = mlpcResultMapping(modelingResults)

    val fittedPipeline = new PostModelingPipelineBuilder(inferenceDataSet)
      .setModelType(modelType)
      .setNumericBoundaries(_numericBoundaries)
      .setStringBoundaries(_stringBoundaries)
      .regressionModelForPermutationTest()

    val fullSearchSpaceDataSet =
      generateMLPCSearchSpaceAsDataFrame(
        featureInputSize,
        classDistinctCount + 1
      ).withColumnRenamed("layers", "layerConstruct")
        .withColumnRenamed("layerCount", "layers")

    val restrictedData =
      transformAndLimit(fittedPipeline, fullSearchSpaceDataSet, topPredictions)
        .withColumnRenamed("layers", "layerCount")
        .withColumnRenamed("layerConstruct", "layers")

    convertMLPCResultToConfig(
      restrictedData,
      featureInputSize,
      classDistinctCount + 1
    )
  }

}
