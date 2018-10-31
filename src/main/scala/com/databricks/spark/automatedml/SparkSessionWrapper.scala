package com.databricks.spark.automatedml

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

trait SparkSessionWrapper {

  lazy val spark: SparkSession = SparkSession
    .builder()
    .appName("Databricks Automated ML")
    .getOrCreate()

  lazy val sc: SparkContext = SparkContext.getOrCreate()

}
