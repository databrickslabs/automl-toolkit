package com.databricks.labs.automl.model.tools.structures

import org.apache.spark.sql.DataFrame

case class TrainSplitReferences(kIndex: Int,
                                data: TrainTestData,
                                paths: TrainTestPaths)

case class TrainTestData(train: DataFrame, test: DataFrame)

case class TrainTestPaths(train: String, test: String)
