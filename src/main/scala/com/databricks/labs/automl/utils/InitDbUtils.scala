package com.databricks.labs.automl.utils

/**
 * This util initializes required Dbutils params so that when invoked from
 * pyspark, it doesn't result in NPE due to runtime proxy injections
 */
object InitDbUtils {

  def getNotebookPath: String = DBUtilsHelper.getNotebookPath
  def getNotebookDirectory: String = DBUtilsHelper.getNotebookDirectory
  def getTrackingURI: String = DBUtilsHelper.getTrackingURI
  def getAPIToken: String = DBUtilsHelper.getAPIToken

  def validate(): Unit = {
    assert(!"".equals(getNotebookPath), "NotebookPath cannot be null")
    assert(!"".equals(getNotebookDirectory, "NotebookDirectory cannot be null"))
    assert(!"".equals(getTrackingURI, "TrackingURI cannot be null"))
    assert(!"".equals(getAPIToken, "APIToken cannot be null"))
  }

}
