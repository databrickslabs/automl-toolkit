package com.databricks.labs.automl.exploration.structures

trait FeatureImportanceTools {

  import com.databricks.labs.automl.exploration.structures.CutoffTypes._
  import com.databricks.labs.automl.exploration.structures.FeatureImportanceModelFamily._
  import com.databricks.labs.automl.exploration.structures.ModelType._

  private[exploration] def cutoffTypeEvaluator(value: String): CutoffTypes = {

    value.toLowerCase.replaceAll("\\s", "") match {
      case "none"  => None
      case "value" => Threshold
      case "count" => Count
      case _ =>
        throw new IllegalArgumentException(
          s"$value is not supported! Must be one of: 'none', 'value', or " +
            s"'count' "
        )
    }
  }

  private[exploration] def featureImportanceFamilyEvaluator(
    value: String
  ): FeatureImportanceModelFamily = {
    value.toLowerCase.replaceAll("\\s", "") match {
      case "randomforest" => RandomForest
      case "xgboost"      => XGBoost
      case _ =>
        throw new IllegalArgumentException(
          s"$value is not supported! Must be either 'RandomForest' or 'XGBoost'"
        )
    }
  }

  private[exploration] def modelTypeEvaluator(value: String): ModelType = {
    value.toLowerCase.replaceAll("\\s", "") match {
      case "regressor"  => Regressor
      case "classifier" => Classifier
      case _ =>
        throw new IllegalArgumentException(
          s"$value is not supported! Must be either 'Regressor' or 'Classifier'"
        )
    }
  }

}
