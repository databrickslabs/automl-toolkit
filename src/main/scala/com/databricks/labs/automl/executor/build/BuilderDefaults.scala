package com.databricks.labs.automl.executor.build

trait BuilderDefaults {

  import FamilyValidator._
  import PredictionType._
  import ModelSelector._
  import ModelDefaults._

  /**
    * General Tools
    */

  private[build] def modelTypeEvaluator(modelFamily: String, predictionType: String): ModelSelector = {
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

  private[build] def predictionTypeEvaluator(predictionType: String): PredictionType = {
    predictionType.toLowerCase.replaceAll("\\s", "") match {
      case "regressor" => Regressor
      case "classifier" => Classifier
      case _ => throw new IllegalArgumentException(s"$predictionType is not a supported type! Must be either: " +
        s"'regressor' or 'classifier'")
    }
  }

  private[build] def familyTypeEvaluator(modelFamily: String): FamilyValidator = {
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

  private def familyScoringCheck(predictionType: PredictionType): String = {
    predictionType match {
      case Regressor => "rmse"
      case _ => "areaUnderROC"
    }
  }

  private def treesBooleanSwitch(modelType: FamilyValidator): Boolean = {
    modelType match {
      case Trees => false
      case _ => true
    }
  }

  def oneHotEncodeFlag(family: FamilyValidator): Boolean = treesBooleanSwitch(family)
  def scalingFlag(family: FamilyValidator): Boolean = treesBooleanSwitch(family)

  private def familyScoringDirection(predictionType: PredictionType): String = {
    predictionType match {
      case Regressor => "minimize"
      case _ => "maximize"
    }
  }

  /**
    * Algorithm Helper Methods
    */

  private[build] def boundaryValidation(modelKeys: Set[String], overwriteKeys: Set[String]): Unit = {
    require(modelKeys == overwriteKeys, s"The provided configuration does not match. Expected: " +
      s"${modelKeys.mkString(", ")}, but got: ${overwriteKeys.mkString(", ")} }")
  }

  private[build] def validateNumericBoundariesKeys(modelType: ModelSelector, value: Map[String, (Double, Double)]): Unit = {
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

  private[build] def validateNumericBoundariesValues(values: Map[String, (Double, Double)]): Unit = {
    values.foreach(k => require(k._2._1 < k._2._2, s"Numeric Boundary key ${k._1} is set incorrectly! " +
      s"Boundary definitions must be in the form: (min, max)"))
  }

  private[build] def numericBoundariesAssignment(modelType: ModelSelector): Map[String, (Double, Double)] = {
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

  private[build] def validateStringBoundariesKeys(modelType: ModelSelector, value: Map[String, List[String]]): Unit = {
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

  private[build] def stringBoundariesAssignment(modelType: ModelSelector): Map[String, List[String]] = {
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

