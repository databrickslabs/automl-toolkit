package com.databricks.labs.automl.utils

/**
 * This util initializes required Dbutils params so that when invoked from
 * pyspark, it doesn't result in NPE due to runtime proxy injections
 */
object InitDbUtils {

  val getNotebookPath: String = DBUtilsHelper.getNotebookPath
  val getNotebookDirectory: String = DBUtilsHelper.getNotebookDirectory
  val getTrackingURI: String = DBUtilsHelper.getTrackingURI
  val getAPIToken: String = DBUtilsHelper.getAPIToken

}
