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

case class GenericConfig(
                        var labelCol: String,
                        var featuresCol: String,
                        var dateTimeConversionType: String,
                        var fieldsToIgnoreInVector: Array[String],
                        var scoringMetric: String,
                        )

class GenericConfigGenerator(predictionType: String) extends BuilderDefaults {

  import PredictionType._

  private val familyType: PredictionType = predictionTypeEvaluator(predictionType)

  private var _genericConfig = genericConfig(familyType)


  def setLabelCol(value: String): this.type = {_genericConfig.labelCol = value; this}
  def setFeaturesCol(value: String): this.type = {_genericConfig.featuresCol = value; this}
  def setDateTimeConversionType(value: String): this.type = {
    require(allowedDateTimeConversionTypes.contains(value), s"Invalid DateTimeConversionType: $value . " +
      s"Must be one of: ${allowedDateTimeConversionTypes.mkString(", ")}")
    _genericConfig.dateTimeConversionType = value
    this
  }
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

  def getLabelCol: String = _genericConfig.labelCol
  def getFeaturesCol: String = _genericConfig.featuresCol
  def getDateTimeConversionType: String = _genericConfig.dateTimeConversionType
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

  private val modelType: ModelSelector = (modelFamily, predictionType) match {
    case ("trees", "regressor") => TreesRegressor
    case ("trees", "classifier") => TreesClassifier
    case ("gbt", "regressor") => GBTRegressor
    case ("gbt", "classifier") => GBTClassifier
    case ("randomForest", "regressor") => RandomForestRegressor
    case ("randomForest", "classifier") => RandomForestClassifier
    case ("linearRegression", "regressor") => LinearRegression
  }

  private val family: FamilyValidator = modelFamily match {
    case "trees" | "gbt" | "randomForest" | "xgBoost" => Trees
    case _ => NonTrees
  }

  private var _switchConfig = switchConfig(family)

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

  def getSwitchConfig: SwitchConfig = _switchConfig

  def getInstanceConfig: InstanceConfig = ???

}


object ConfigurationGenerator extends BuilderDefaults {

  import PredictionType._

  def apply(modelFamily: String, predictionType: String, genericConfig: GenericConfig): ConfigurationGenerator =
    new ConfigurationGenerator(modelFamily, predictionType, genericConfig)

  def generateDefaultConfig(modelFamily: String, predictionType: String): InstanceConfig = {

    val familyType = predictionTypeEvaluator(predictionType)

    familyType match {
      case Regressor => new ConfigurationGenerator(modelFamily, predictionType,
        GenericConfigGenerator.generateDefaultRegressorConfig).getInstanceConfig
      case Classifier => new ConfigurationGenerator(modelFamily, predictionType,
        GenericConfigGenerator.generateDefaultClassifierConfig).getInstanceConfig
    }

  }

}


case class InstanceConfig(
                     modelFamily: String,
                     predictionType: String,
                     switchConfig: SwitchConfig,
                     featureEngineeringConfig: FeatureEngineeringConfig,
                     algorithmConfig: AlgorithmConfig,
                     tunerConfig: TunerConfig,
                     loggingConfig: LoggingConfig
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

case class FeatureEngineeringConfig(
                        labelCol: String,
                        featuresCol: String
                        )
case class TunerConfig(

                      )
case class AlgorithmConfig(
                          stringBoundaries: Map[String, List[String]],
                          numericBoundaries: Map[String, (Double, Double)]
                          )
case class LoggingConfig(

                        )


