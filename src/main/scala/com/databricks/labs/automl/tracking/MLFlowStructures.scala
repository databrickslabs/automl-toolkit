package com.databricks.labs.automl.tracking

import org.mlflow.tracking.MlflowClient

case class MLFlowReturn(client: MlflowClient,
                        experimentId: String,
                        runIdPayload: Array[(String, Double)])

case class MLFlowReportStructure(fullLog: MLFlowReturn, bestLog: MLFlowReturn)
