package com.databricks.spark.automatedml.utils

object ModelFamily extends Enumeration {

  val RANDOM_FOREST = Value("RandomForest")

}

object ModelType extends Enumeration {

  val CLASSIFIER = Value("classifier")

}