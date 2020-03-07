package com.databricks.labs.automl.utils

import java.nio.file.Paths

/**
 * This util initializes required Dbutils params so that when invoked from
 * pyspark, it doesn't result in NPE due to runtime proxy injections
 */
object InitDbUtils {

  case class LogggingConfigType(mlFlowTrackingURI: String, mlFlowExperimentName: String, mlFlowAPIToken: String, mlFlowModelSaveDirectory: String)

  def getNotebookPath: String = DBUtilsHelper.getNotebookPath
  def getNotebookDirectory: String = DBUtilsHelper.getNotebookDirectory
  def getTrackingURI: String = DBUtilsHelper.getTrackingURI
  def getAPIToken: String = DBUtilsHelper.getAPIToken

  def validate(): Unit = {
    assert(getNotebookPath != null && !"".equals(getNotebookPath), "NotebookPath cannot be null")
    assert(getNotebookDirectory != null && !"".equals(getNotebookDirectory, "NotebookDirectory cannot be null"))
    assert(getTrackingURI != null && !"".equals(getTrackingURI, "TrackingURI cannot be null"))
    assert(getAPIToken != null && !"".equals(getAPIToken, "APIToken cannot be null"))
  }

  def getMlFlowLoggingConfig(mlFlowLoggingFlag: Boolean): LogggingConfigType = {
    if(mlFlowLoggingFlag) {
      validate()
      LogggingConfigType(
        getTrackingURI,
        Paths.get(InitDbUtils.getNotebookDirectory + "/MLFlowLogs" ).toString,
        InitDbUtils.getAPIToken,
        Paths.get(InitDbUtils.getNotebookDirectory + "/AutoML_Artifacts").toString
      )
    } else {
      LogggingConfigType("http://localhost:5000/", "/tmp/local_mlflow_exp", "", "/tmp/local_mlflow_exp/artifacts")
    }
  }
}
