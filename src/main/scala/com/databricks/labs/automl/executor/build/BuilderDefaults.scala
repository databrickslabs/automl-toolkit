package com.databricks.labs.automl.executor.build


trait BuilderDefaults {

  import FamilyValidator._
  import PredictionType._

  def predictionTypeEvaluator(predictionType: String): PredictionType = {
    predictionType match {
      case "regressor" => Regressor
      case "classifier" => Classifier
      case _ => throw new IllegalArgumentException(s"$predictionType is not a supported type! Must be either: " +
        s"'regressor' or 'classifier'")
    }
  }

  /**
    * Generic configurations
    */

  final val allowedDateTimeConversionTypes: List[String] = List("unix", "split")
  final val allowableRegressionScoringMetrics: List[String] =  List("rmse", "mse", "r2", "mae")
  final val allowableClassificationScoringMetrics: List[String] = List("f1", "weightedPrecision", "weightedRecall",
    "accuracy", "areaUnderPR", "areaUnderROC")



  def genericConfig(predictionType: PredictionType): GenericConfig = GenericConfig( "label", "features", "split",
    Array.empty[String], familyScoringCheck(predictionType))

  def switchConfig(family: FamilyValidator): SwitchConfig = SwitchConfig(true, true, false, false, false,
    oneHotEncodeFlag(family), scalingFlag(family), true, false)

  private def familyScoringCheck(predictionType: PredictionType): String = {
    predictionType match {
      case Regressor => "rmse"
      case Classifier => "areaUnderROC"
      case a:_ => throw new IllegalArgumentException(s"${a.toString} is not a supported prediction type.")
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

  /**
    * Model Specific configurations
    */

}