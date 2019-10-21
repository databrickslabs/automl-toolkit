package com.databricks.labs.automl.utils

/**
 * This util initializes required Dbutils params so that when invoked from
 * pyspark, doesn't result into NPE due to runtime proxy injection
 */
object InitDbUtils {

  val getNotebookPath: String = DBUtilsHelper.getNotebookPath
  val getNotebookDirectory: String = DBUtilsHelper.getNotebookDirectory
  val getTrackingURI: String = DBUtilsHelper.getTrackingURI
  val getAPIToken: String = DBUtilsHelper.getAPIToken

}
