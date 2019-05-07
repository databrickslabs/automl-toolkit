package com.databricks.spark.automatedml.executor.build

trait BuilderConfig {

  final val classifierModels: List[String] = List("Trees", "RandomForest", "GBT", "LogisticRegression", "MLPC",
    "XGBoost")

  final val regressorModels: List[String] = List("Trees", "RandomForest", "GBT", "LinearRegression", "SVM", "XGBoost")

  final val allowableModelTypes: List[String] = List("classifier", "regressor")








}
