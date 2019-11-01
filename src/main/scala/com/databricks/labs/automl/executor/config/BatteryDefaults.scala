package com.databricks.labs.automl.executor.config

import com.databricks.labs.automl.executor.config.ModelSelector.ModelSelector
import com.databricks.labs.automl.executor.config.PredictionType.PredictionType

trait BatteryDefaults {

  /**
    * Default constructor for a given family of models to generate enumeration types of models to execute
    * @param predictionType Supplied prediction type (either `Regressor` or `Classifier`
    * @return An array of models that support the provided predictionType supplied in the constructor.
    * @since 0.5.0.3
    */
  def modelSelection(predictionType: PredictionType): Array[ModelSelector] = {

    predictionType match {
      case PredictionType.Regressor =>
        RegressorModels.values.toArray
          .map(_.toString)
          .map(x => ModelSelector.withName(x))
      case PredictionType.Classifier =>
        ClassiferModels.values.toArray
          .map(_.toString)
          .map(x => ModelSelector.withName(x))
      case _ =>
        throw new UnsupportedOperationException(
          s"PrecitionType ${predictionType.toString} is not a supported" +
            s"type.  Must be one of: ${PredictionType.values.mkString(", ")}"
        )
    }
  }

  def modelToStandardizedString(modelType: ModelSelector): String = {

    modelType match {
      case ModelSelector.GBTClassifier          => "gbt"
      case ModelSelector.GBTRegressor           => "gbt"
      case ModelSelector.LinearRegression       => "linearregression"
      case ModelSelector.LogisticRegression     => "logisticregression"
      case ModelSelector.MLPC                   => "mlpc"
      case ModelSelector.RandomForestClassifier => "randomforest"
      case ModelSelector.RandomForestRegressor  => "randomforest"
      case ModelSelector.SVM                    => "svm"
      case ModelSelector.TreesClassifier        => "trees"
      case ModelSelector.TreesRegressor         => "trees"
      case ModelSelector.XGBoostClassifier      => "xgboost"
      case ModelSelector.XGBoostRegressor       => "xgboost"
      case ModelSelector.LightGBMBinary         => "gbmbinary"
      case ModelSelector.LightGBMMulti          => "gbmmulti"
      case ModelSelector.LightGBMMultiOVA       => "gbmmultiova"
      case ModelSelector.LightGBMHuber          => "gbmhuber"
      case ModelSelector.LightGBMFair           => "gbmfair"
      case ModelSelector.LightGBMLasso          => "gbmlasso"
      case ModelSelector.LightGBMRidge          => "gbmridge"
      case ModelSelector.LightGBMPoisson        => "gbmpoisson"
      case ModelSelector.LightGBMQuantile       => "gbmquantile"
      case ModelSelector.LightGBMMape           => "gbmmape"
      case ModelSelector.LightGBMTweedie        => "gbmtweedie"
      case ModelSelector.LightGBMGamma          => "gbmgamma"
      case _ =>
        throw new UnsupportedOperationException(
          s"'${modelType.toString}' is not supported."
        )
    }

  }

}
