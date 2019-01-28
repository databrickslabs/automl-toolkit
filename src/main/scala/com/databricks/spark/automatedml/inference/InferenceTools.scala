package com.databricks.spark.automatedml.inference

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import java.io._

trait InferenceTools {


  def createInferencePayload(data: DataFrame, modelingColumns: Array[String], allColumns: Array[String]):
  InferencePayload = {
    new InferencePayload {
      override def data: DataFrame = data
      override def modelingColumns: Array[String] = modelingColumns
      override def allColumns: Array[String] = allColumns
    }
  }

  def removeArrayOfColumns(payload: InferencePayload, removalArray: Array[String]): InferencePayload = {

    val featureRemoval = payload.modelingColumns.diff(removalArray)
    val fullRemoval = payload.allColumns.diff(removalArray)
    val data = payload.data.select(fullRemoval map col:_*)

    createInferencePayload(data, featureRemoval, fullRemoval)

  }


  def saveInferenceConfig(config: InferenceMainConfig): Unit = {

    val outputWriter = new ObjectOutputStream(new FileOutputStream(config.inferenceConfigStorageLocation))

    outputWriter.writeObject(config)
    outputWriter.close()

  }

  def loadInferenceConfig(path: String): InferenceMainConfig = {

    val inputReader = new ObjectInputStream(new FileInputStream(path))
    val storedConfig = inputReader.readObject()
    inputReader.close()

    storedConfig.asInstanceOf[InferenceMainConfig]

  }

}
