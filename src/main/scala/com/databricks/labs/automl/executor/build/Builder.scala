package com.databricks.labs.automl.executor.build


object ModelSelector extends Enumeration {
  type ModelSelector = Value
  val TreesRegressor, TreesClassifier, GBTRegressor, GBTClassifier, LinearRegression, LogisticRegression, MLPC,
  RandomForestRegressor, RandomForestClassifier, SVM, XGBoostRegressor, XGBoostClassifier = Value
}

object FamilyValidator extends Enumeration {
  type FamilyValidator = Value
  val Trees, NonTrees = Value
}

object PredictionType extends Enumeration {
  type PredictionType = Value
  val Regressor, Classifier = Value
}



class GenericConfigGenerator(predictionType: String) extends BuilderDefaults {

  import PredictionType._

  private val familyType: PredictionType = predictionTypeEvaluator(predictionType)

  private var _genericConfig = genericConfig(familyType)


  def setLabelCol(value: String): this.type = {_genericConfig.labelCol = value; this}
  def setFeaturesCol(value: String): this.type = {_genericConfig.featuresCol = value; this}
  def setDateTimeConversionType(value: String): this.type = {
    require(allowableDateTimeConversionTypes.contains(value), s"Invalid DateTimeConversionType: $value . " +
      s"Must be one of: ${allowableDateTimeConversionTypes.mkString(", ")}")
    _genericConfig.dateTimeConversionType = value
    this
  }
  def setFieldsToIgnoreInVector(value: Array[String]): this.type = {_genericConfig.fieldsToIgnoreInVector = value; this}
  def setScoringMetric(value: String): this.type = {
    familyType match {
      case Regressor => require(allowableRegressionScoringMetrics.contains(value), s"Prediction family type " +
        s"$predictionType scoring metric $value is not supported.  Must be one of: " +
        s"${allowableRegressionScoringMetrics.mkString(", ")}")
      case Classifier => require(allowableClassificationScoringMetrics.contains(value), s"Prediction family type " +
        s"$predictionType scoring metric $value is not supported.  Must be one of: " +
        s"${allowableClassificationScoringMetrics.mkString(", ")}")
    }
    _genericConfig.scoringMetric = value
    this
  }
  def setScoringOptimizationStrategy(value: String): this.type = {
    require(allowableScoringOptimizationStrategies.contains(value), s"Scoring Optimization Strategy $value is not " +
      s"supported.  Must be one of: ${allowableScoringOptimizationStrategies.mkString(", ")}")
    _genericConfig.scoringOptimizationStrategy = value
    this
  }

  def getLabelCol: String = _genericConfig.labelCol
  def getFeaturesCol: String = _genericConfig.featuresCol
  def getDateTimeConversionType: String = _genericConfig.dateTimeConversionType
  def getFieldsToIgnoreInVector: Array[String] = _genericConfig.fieldsToIgnoreInVector
  def getScoringMetric: String = _genericConfig.scoringMetric
  def getScoringOptimizationStrategy: String = _genericConfig.scoringOptimizationStrategy
  def getConfig: GenericConfig = _genericConfig

}

object GenericConfigGenerator {

  def generateDefaultClassifierConfig: GenericConfig = new GenericConfigGenerator("classifier").getConfig

  def generateDefaultRegressorConfig: GenericConfig = new GenericConfigGenerator("regressor").getConfig
}

class ConfigurationGenerator(modelFamily: String, predictionType: String, genericConfig: GenericConfig)
  extends BuilderDefaults {

  import ModelSelector._
  import FamilyValidator._

  private val modelType: ModelSelector = modelTypeEvaluator(modelFamily, predictionType)
  private val family: FamilyValidator = familyTypeEvaluator(modelFamily)

  /**
    * Default configuration generation
    */

  private var _switchConfig = switchConfig(family)
  private var _algorithmConfig = algorithmConfig(modelType)
  private var _featureEngineeringConfig = featureEngineeringConfig()


  private var _instanceConfig = instanceConfig(modelFamily, predictionType)

  /**
    * Switch Config
    */

  def naFillOn(): this.type = {_switchConfig.naFillFlag = true; this}
  def naFillOff(): this.type = {_switchConfig.naFillFlag = false; this}
  def varianceFilterOn(): this.type = {_switchConfig.varianceFilterFlag = true; this}
  def varianceFilterOff(): this.type = {_switchConfig.varianceFilterFlag = false; this}
  def outlierFilterOn(): this.type = {_switchConfig.outlierFilterFlag = true; this}
  def outlierFilterOff(): this.type = {_switchConfig.outlierFilterFlag = false; this}
  def pearsonFilterOn(): this.type = {_switchConfig.pearsonFilterFlag = true; this}
  def pearsonFilterOff(): this.type = {_switchConfig.pearsonFilterFlag = false; this}
  def covarianceFilterOn(): this.type = {_switchConfig.covarianceFilterFlag = true; this}
  def covarianceFilterOff(): this.type = {_switchConfig.covarianceFilterFlag = false; this}
  def oneHotEncodeOn(): this.type = {
    family match {
      case Trees => println("WARNING! OneHotEncoding set on a trees algorithm will likely create a poor model.  " +
        "Proceed at your own risk!")
    }
    _switchConfig.oneHotEncodeFlag = true
    this
  }
  def oneHotEncodeOff(): this.type = {_switchConfig.oneHotEncodeFlag = false; this}
  def scalingOn(): this.type = {_switchConfig.scalingFlag = true; this}
  def scalingOff(): this.type = {_switchConfig.scalingFlag = false; this}
  def dataPrepCachingOn(): this.type = {_switchConfig.dataPrepCachingFlag = true; this}
  def dataPrepCachingOff(): this.type = {_switchConfig.dataPrepCachingFlag = false; this}
  def autoStoppingOn(): this.type = {_switchConfig.autoStoppingFlag = true; this}
  def autoStoppingOff(): this.type = {_switchConfig.autoStoppingFlag = false; this}


  /**
    * Feature Engineering Config
    */

  def setFillConfigNumericFillStat(value: String): this.type = {
    require(allowableNumericFillStats.contains(value), s"Numeric Fill Stat $value is not supported.  Must be one of:" +
      s"${allowableNumericFillStats.mkString(", ")}")
    _featureEngineeringConfig.numericFillStat = value
    this
  }
  def setFillConfigCharacterFillStat(value: String): this.type = {
    require(allowableCharacterFillStats.contains(value), s"Character Fill Stat $value is not supported.  Must be one " +
      s"of: ${allowableCharacterFillStats.mkString(", ")}")
    _featureEngineeringConfig.characterFillStat = value
    this
  }
  def setFillConfigmodelSelectionDistinctThreshold(value: Int): this.type = { //TODO: warning
    ???
    this
  }
  def setOutlierFilterBounds(value: String): this.type = {
    ???
    this
  }
  def setOutlierLowerFilterNTile(value: Double): this.type = {
    ???
    this
  }
  def setOutlierUpperFilterNTile(value: Double): this.type = {
    ???
    this
  }
  def setOutlierFilterPrecision(value: Double): this.type = {
    ???
    this
  }
  def setOutlierContinuousDataThreshold(value: Int): this.type = {
    ???
    this
  }

  /**
    * Algorithm Config
    */

  def setStringBoundaries(value: Map[String, List[String]]): this.type = {
    validateStringBoundariesKeys(modelType, value)
    _algorithmConfig.stringBoundaries = value
    this
  }

  def setNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    validateNumericBoundariesValues(value)
    validateNumericBoundariesKeys(modelType, value)
    _algorithmConfig.numericBoundaries = value
    this
  }

  /**
    * Tuner Config
    */

  /**
    * MLFlow Logging Config
    */

  /**
    * Getters
    */

  def getSwitchConfig: SwitchConfig = _switchConfig

  def getAlgorithmConfig: AlgorithmConfig = _algorithmConfig

  def getFeatureEngineeringConfig: FeatureEngineeringConfig = _featureEngineeringConfig

  def getInstanceConfig: InstanceConfig = _instanceConfig

  //TODO : json input and extract for case class definitions
  //TODO: implicit reflection for map type config?


}


object ConfigurationGenerator extends BuilderDefaults {

  import PredictionType._

  def apply(modelFamily: String, predictionType: String, genericConfig: GenericConfig): ConfigurationGenerator =
    new ConfigurationGenerator(modelFamily, predictionType, genericConfig)

  def generateDefaultConfig(modelFamily: String, predictionType: String): InstanceConfig = {

    predictionTypeEvaluator(predictionType) match {
      case Regressor => new ConfigurationGenerator(modelFamily, predictionType,
        GenericConfigGenerator.generateDefaultRegressorConfig).getInstanceConfig
      case Classifier => new ConfigurationGenerator(modelFamily, predictionType,
        GenericConfigGenerator.generateDefaultClassifierConfig).getInstanceConfig
    }

  }

}

case class GenericConfig(
                          var labelCol: String,
                          var featuresCol: String,
                          var dateTimeConversionType: String,
                          var fieldsToIgnoreInVector: Array[String],
                          var scoringMetric: String,
                          var scoringOptimizationStrategy: String
                        )

case class FeatureEngineeringConfig(
                                     var numericFillStat: String,
                                     var characterFillStat: String,
                                     var modelSelectionDistinctThreshold: Int,
                                     var outlierFilterBounds: String,
                                     var outlierLowerFilterNTile: Double,
                                     var outlierUpperFilterNTile: Double,
                                     var outlierFilterPrecision: Double,
                                     var outlierContinuousDataThreshold: Int,
                                     var outlierFieldsToIgnore: Array[String],
                                     var pearsonFilterStatistic: String,
                                     var pearsonFilterDirection: String,
                                     var pearsonFilterManualValue: Double,
                                     var pearsonFilterMode: String,
                                     var pearsonAutoFilterNTile: Double,
                                     var covarianceCorrelationCutoffLow: Double,
                                     var covarianceCorrelationCutoffHigh: Double,
                                     var scalingType: String,
                                     var scalingMin: Double,
                                     var scalingMax: Double,
                                     var scalingStandardMeanFlag: Boolean,
                                     var scalingStdDevFlag: Boolean,
                                     var scalingPNorm: Double
                                   )

case class SwitchConfig(
                         var naFillFlag: Boolean,
                         var varianceFilterFlag: Boolean,
                         var outlierFilterFlag: Boolean,
                         var pearsonFilterFlag: Boolean,
                         var covarianceFilterFlag: Boolean,
                         var oneHotEncodeFlag: Boolean,
                         var scalingFlag: Boolean,
                         var dataPrepCachingFlag: Boolean,
                         var autoStoppingFlag: Boolean
                       )


case class TunerConfig(
                        var autoStoppingScore: Double //
                      )
case class AlgorithmConfig(
                            var stringBoundaries: Map[String, List[String]],
                            var numericBoundaries: Map[String, (Double, Double)]
                          )
case class LoggingConfig(

                        )


case class InstanceConfig(
                     var modelFamily: String,
                     var predictionType: String,
                     var genericConfig: GenericConfig,
                     var switchConfig: SwitchConfig,
                     var featureEngineeringConfig: FeatureEngineeringConfig,
                     var algorithmConfig: AlgorithmConfig,
                     var tunerConfig: TunerConfig,
                     var loggingConfig: LoggingConfig
                     )





trait BuilderDefaults {

  import FamilyValidator._
  import PredictionType._
  import ModelSelector._
  import ModelDefaults._

  /**
    * General Tools
    */

  def modelTypeEvaluator(modelFamily: String, predictionType: String): ModelSelector = {
    (modelFamily.toLowerCase.replaceAll("\\s", ""),
      predictionType.toLowerCase.replaceAll("\\s", "")) match {
      case ("trees", "regressor") => TreesRegressor
      case ("trees", "classifier") => TreesClassifier
      case ("gbt", "regressor") => GBTRegressor
      case ("gbt", "classifier") => GBTClassifier
      case ("randomforest", "regressor") => RandomForestRegressor
      case ("randomforest", "classifier") => RandomForestClassifier
      case ("linearregression", "regressor") => LinearRegression
      case ("logisticregression", "classifier") => LogisticRegression
      case ("xgboost", "regressor") => XGBoostRegressor
      case ("xgboost", "classifier") => XGBoostClassifier
      case ("mlpc", "classifier") => MLPC
      case ("svm", "regressor") => SVM
      case (_,_) => throw new IllegalArgumentException(s"$modelFamily Model Family and $predictionType are not supported.")
    }
  }

  def predictionTypeEvaluator(predictionType: String): PredictionType = {
    predictionType.toLowerCase.replaceAll("\\s", "") match {
      case "regressor" => Regressor
      case "classifier" => Classifier
      case _ => throw new IllegalArgumentException(s"$predictionType is not a supported type! Must be either: " +
        s"'regressor' or 'classifier'")
    }
  }

  def familyTypeEvaluator(modelFamily: String): FamilyValidator = {
    modelFamily.toLowerCase.replaceAll("\\s", "") match {
      case "trees" | "gbt" | "randomforest" | "xgboost" => Trees
      case _ => NonTrees
    }
  }


  /**
    * Static restrictions
    */

  final val allowableDateTimeConversionTypes: List[String] = List("unix", "split")
  final val allowableRegressionScoringMetrics: List[String] =  List("rmse", "mse", "r2", "mae")
  final val allowableClassificationScoringMetrics: List[String] = List("f1", "weightedPrecision", "weightedRecall",
    "accuracy", "areaUnderPR", "areaUnderROC")
  final val allowableScoringOptimizationStrategies: List[String] = List("minimize", "maximize")
  final val allowableNumericFillStats: List[String] = List("min", "25p", "mean", "median", "75p", "max")
  final val allowableCharacterFillStats: List[String] = List("min", "max")



  /**
    * Generic Helper Methods
    */

  def familyScoringCheck(predictionType: PredictionType): String = {
    predictionType match {
      case Regressor => "rmse"
      case _ => "areaUnderROC"
    }
  }

  def treesBooleanSwitch(modelType: FamilyValidator): Boolean = {
    modelType match {
      case Trees => false
      case _ => true
    }
  }

  def oneHotEncodeFlag(family: FamilyValidator): Boolean = treesBooleanSwitch(family)
  def scalingFlag(family: FamilyValidator): Boolean = treesBooleanSwitch(family)

  def familyScoringDirection(predictionType: PredictionType): String = {
    predictionType match {
      case Regressor => "minimize"
      case _ => "maximize"
    }
  }

  /**
    * Algorithm Helper Methods
    */

  def boundaryValidation(modelKeys: Set[String], overwriteKeys: Set[String]): Unit = {
    require(modelKeys == overwriteKeys, s"The provided configuration does not match. Expected: " +
      s"${modelKeys.mkString(", ")}, but got: ${overwriteKeys.mkString(", ")} }")
  }

  def validateNumericBoundariesKeys(modelType: ModelSelector, value: Map[String, (Double, Double)]): Unit = {
    modelType match {
      case RandomForestRegressor => boundaryValidation(randomForestNumeric.keys.toSet, value.keys.toSet)
      case RandomForestClassifier => boundaryValidation(randomForestNumeric.keys.toSet, value.keys.toSet)
      case TreesRegressor => boundaryValidation(treesNumeric.keys.toSet, value.keys.toSet)
      case TreesClassifier => boundaryValidation(treesNumeric.keys.toSet, value.keys.toSet)
      case XGBoostRegressor => boundaryValidation(xgBoostNumeric.keys.toSet, value.keys.toSet)
      case XGBoostClassifier => boundaryValidation(xgBoostNumeric.keys.toSet, value.keys.toSet)
      case MLPC => boundaryValidation(mlpcNumeric.keys.toSet, value.keys.toSet)
      case GBTRegressor => boundaryValidation(gbtNumeric.keys.toSet, value.keys.toSet)
      case GBTClassifier => boundaryValidation(gbtNumeric.keys.toSet, value.keys.toSet)
      case LinearRegression => boundaryValidation(linearRegressionNumeric.keys.toSet, value.keys.toSet)
      case LogisticRegression => boundaryValidation(logisticRegressionNumeric.keys.toSet, value.keys.toSet)
      case SVM => boundaryValidation(svmNumeric.keys.toSet, value.keys.toSet)
    }
  }

  def validateNumericBoundariesValues(values: Map[String, (Double, Double)]): Unit = {
    values.foreach(k => require(k._2._1 < k._2._2, s"Numeric Boundary key ${k._1} is set incorrectly! " +
      s"Boundary definitions must be in the form: (min, max)"))
  }

  def numericBoundariesAssignment(modelType: ModelSelector): Map[String, (Double, Double)] = {
    modelType match {
      case RandomForestRegressor => randomForestNumeric
      case RandomForestClassifier => randomForestNumeric
      case TreesRegressor => treesNumeric
      case TreesClassifier => treesNumeric
      case XGBoostRegressor => xgBoostNumeric
      case XGBoostClassifier => xgBoostNumeric
      case MLPC => mlpcNumeric
      case GBTRegressor => gbtNumeric
      case GBTClassifier => gbtNumeric
      case LinearRegression => linearRegressionNumeric
      case LogisticRegression => logisticRegressionNumeric
      case SVM => svmNumeric
      case _ => throw new NotImplementedError(s"Model Type ${modelType.toString} is not implemented.")
    }
  }

  def validateStringBoundariesKeys(modelType: ModelSelector, value: Map[String, List[String]]): Unit = {
    modelType match {
      case RandomForestRegressor => boundaryValidation(randomForestString.keys.toSet, value.keys.toSet)
      case RandomForestClassifier => boundaryValidation(randomForestString.keys.toSet, value.keys.toSet)
      case TreesRegressor => boundaryValidation(treesString.keys.toSet, value.keys.toSet)
      case TreesClassifier => boundaryValidation(treesString.keys.toSet, value.keys.toSet)
      case MLPC => boundaryValidation(mlpcString.keys.toSet, value.keys.toSet)
      case GBTRegressor => boundaryValidation(gbtString.keys.toSet, value.keys.toSet)
      case GBTClassifier => boundaryValidation(gbtString.keys.toSet, value.keys.toSet)
      case LinearRegression => boundaryValidation(linearRegressionString.keys.toSet, value.keys.toSet)
      case _ => throw new IllegalArgumentException(s"${modelType.toString} has no StringBoundaries to configure.")
    }
  }

  def stringBoundariesAssignment(modelType: ModelSelector): Map[String, List[String]] = {
    modelType match {
      case RandomForestRegressor => randomForestString
      case RandomForestClassifier => randomForestString
      case TreesRegressor => treesString
      case TreesClassifier => treesString
      case XGBoostRegressor => Map.empty
      case XGBoostClassifier => Map.empty
      case MLPC => mlpcString
      case GBTRegressor => gbtString
      case GBTClassifier => gbtString
      case LinearRegression => linearRegressionString
      case LogisticRegression => Map.empty
      case SVM => Map.empty
      case _ => throw new NotImplementedError(s"Model Type ${modelType.toString} is not implemented.")
    }
  }

  /**
    * Generate the default configuration objects
    */

  def genericConfig(predictionType: PredictionType): GenericConfig = GenericConfig( "label", "features", "split",
    Array.empty[String], familyScoringCheck(predictionType), familyScoringDirection(predictionType))

  def switchConfig(family: FamilyValidator): SwitchConfig = SwitchConfig(true, true, false, false, false,
    oneHotEncodeFlag(family), scalingFlag(family), true, false)

  def algorithmConfig(modelType: ModelSelector): AlgorithmConfig = AlgorithmConfig(
    stringBoundariesAssignment(modelType), numericBoundariesAssignment(modelType))

  def featureEngineeringConfig(): FeatureEngineeringConfig = FeatureEngineeringConfig(
    "mean", "max", 50,
    "both", 0.02, 0.98, 0.01, 50, Array.empty[String],
    "pearsonStat", "greater", 0.0, "auto", 0.75,
    -0.99, 0.99,
    "minMax", 0.0, 1.0, false, true, 2.0
  )

  // TODO: finish this out properly.
  def instanceConfig(modelFamily: String, predictionType: String): InstanceConfig = {
    val modelingType = predictionTypeEvaluator(predictionType)
    val family = familyTypeEvaluator(modelFamily)
    val modelType = modelTypeEvaluator(modelFamily, predictionType)
    InstanceConfig(
      modelFamily, predictionType, genericConfig(modelingType), switchConfig(family), featureEngineeringConfig(),
      algorithmConfig(modelType), TunerConfig(0.0), LoggingConfig()

    )
  }

  /**
    * Model Specific configurations
    */

}


object ModelDefaults {

def randomForestNumeric: Map[String, (Double, Double)] = Map(
    "numTrees" -> Tuple2(50.0, 1000.0),
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "subSamplingRate" -> Tuple2(0.5, 1.0)
  )

def randomForestString: Map[String, List[String]] = Map(
    "impurity" -> List("gini", "entropy"),
    "featureSubsetStrategy" -> List("auto")
  )

def treesNumeric: Map[String, (Double, Double)] = Map(
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "minInstancesPerNode" -> Tuple2(1.0, 50.0)
  )

 def treesString: Map[String, List[String]] = Map(
    "impurity" -> List("gini", "entropy")
  )

 def xgBoostNumeric: Map[String, (Double, Double)] = Map(
    "alpha" -> Tuple2(0.0, 1.0),
    "eta" -> Tuple2(0.1, 0.5),
    "gamma" -> Tuple2(0.0, 10.0),
    "lambda" -> Tuple2(0.1, 10.0),
    "maxDepth" -> Tuple2(3.0, 10.0),
    "subSample" -> Tuple2(0.4, 0.6),
    "minChildWeight" -> Tuple2(0.1, 10.0),
    "numRound" -> Tuple2(5.0, 25.0),
    "maxBins" -> Tuple2(25.0, 512.0),
    "trainTestRatio" -> Tuple2(0.2, 0.8)
  )
 def mlpcNumeric: Map[String, (Double, Double)] = Map(
    "layers" -> Tuple2(1.0, 10.0),
    "maxIter" -> Tuple2(10.0, 100.0),
    "stepSize" -> Tuple2(0.01, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5),
    "hiddenLayerSizeAdjust" -> Tuple2(0.0, 50.0)
  )

 def mlpcString: Map[String, List[String]] = Map(
    "solver" -> List("gd", "l-bfgs")
  )

 def gbtNumeric: Map[String, (Double, Double)] = Map(
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxIter" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "minInstancesPerNode" -> Tuple2(1.0, 50.0),
    "stepSize" -> Tuple2(1E-4, 1.0)
  )

 def gbtString: Map[String, List[String]] = Map(
    "impurity" -> List("gini", "entropy"),
    "lossType" -> List("logistic")
  )
def linearRegressionNumeric: Map[String, (Double, Double)] = Map(
    "elasticNetParams" -> Tuple2(0.0, 1.0),
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5)
  )
def linearRegressionString: Map[String, List[String]] = Map (
    "loss" -> List("squaredError", "huber")
  )

def logisticRegressionNumeric: Map[String, (Double, Double)] = Map(
    "elasticNetParams" -> Tuple2(0.0, 1.0),
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5)
  )

  def svmNumeric: Map[String, (Double, Double)] = Map(
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5)
  )

}

