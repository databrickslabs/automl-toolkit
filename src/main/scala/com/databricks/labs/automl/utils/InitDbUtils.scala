package com.databricks.labs.automl.utils

import java.nio.file.Paths

/**
  * This util initializes required Dbutils params so that when invoked from
  * pyspark, it doesn't result in NPE due to runtime proxy injections
  */
object InitDbUtils {

  case class LoggingConfigType(mlFlowTrackingURI: String,
                               mlFlowExperimentName: String,
                               mlFlowAPIToken: String,
                               mlFlowModelSaveDirectory: String)

  final private val DEFAULT_REMOTE_LOGGING_PATH: String =
    "dbfs:/ml/automl/AutoML_Artifacts"
  final private val DEFAULT_LOCAL_LOGGING_PATH: String =
    "/tmp/local_mlflow_exp/artifacts"
  final private val DEFAULT_MLFLOW_LOGGING_LOCATION: String = "/MLFlowLogs"

  def getNotebookPath: String = DBUtilsHelper.getNotebookPath
  def getNotebookDirectory: String = DBUtilsHelper.getNotebookDirectory
  def getTrackingURI: String = DBUtilsHelper.getTrackingURI
  def getAPIToken: String = DBUtilsHelper.getAPIToken

  def validate(): Unit = {
    assert(
      getNotebookPath != null && !"".equals(getNotebookPath),
      "NotebookPath cannot be null"
    )
    assert(
      getNotebookDirectory != null && !""
        .equals(getNotebookDirectory, "NotebookDirectory cannot be null")
    )
    assert(
      getTrackingURI != null && !""
        .equals(getTrackingURI, "TrackingURI cannot be null")
    )
    assert(
      getAPIToken != null && !"".equals(getAPIToken, "APIToken cannot be null")
    )
  }

  def getMlFlowLoggingConfig(mlFlowLoggingFlag: Boolean): LoggingConfigType = {
    if (mlFlowLoggingFlag) {
      validate()
      LoggingConfigType(
        getTrackingURI,
        Paths
          .get(
            InitDbUtils.getNotebookDirectory + DEFAULT_MLFLOW_LOGGING_LOCATION
          )
          .toString,
        InitDbUtils.getAPIToken,
        Paths.get(DEFAULT_REMOTE_LOGGING_PATH).toString
      )
    } else {
      LoggingConfigType(
        "http://localhost:5000/",
        "/tmp/local_mlflow_exp",
        "",
        DEFAULT_LOCAL_LOGGING_PATH
      )
    }
  }
}
