package com.databricks.labs.automl.inference

import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization._

trait InferenceTools extends SparkSessionWrapper {


  //TODO: for chained feature importances, strip out the _si and _oh from field names.
  /**
    *
    * @param dataFrame
    * @param modelingColumnsPayload
    * @param allColumnsPayload
    * @return
    */
  def createInferencePayload(dataFrame: DataFrame, modelingColumnsPayload: Array[String], allColumnsPayload: Array[String]):
  InferencePayload = {
    new InferencePayload {
      override def data: DataFrame = dataFrame
      override def modelingColumns: Array[String] = modelingColumnsPayload
      override def allColumns: Array[String] = allColumnsPayload
    }
  }

  /**
    *
    * @param payload
    * @param removalArray
    * @return
    */
  def removeArrayOfColumns(payload: InferencePayload, removalArray: Array[String]): InferencePayload = {

    val featureRemoval = payload.modelingColumns.diff(removalArray)
    val fullRemoval = payload.allColumns.diff(removalArray)
    val data = payload.data.select(fullRemoval map col:_*)

    createInferencePayload(data, featureRemoval, fullRemoval)

  }

  /**
    * Handler method for converting the InferenceMainConfig object to a serializable Json String with correct
    * scala-compatible data structures.
    * @param config instance of InferenceMainConfig
    * @return [InferenceJsonReturn] consisting of compact form (for logging) and prettyprint form (human readable)
    */
  def convertInferenceConfigToJson(config: InferenceMainConfig): InferenceJsonReturn = {

    implicit val formats: Formats = Serialization.formats(hints=NoTypeHints)
    val pretty = writePretty(config)
    val compact = write(config)

    InferenceJsonReturn(
      compactJson = compact,
      prettyJson = pretty
    )
  }

  /**
    * Handler method for converting a read-in json config String to an instance of InferenceMainConfig
    * @param jsonConfig the config as a Json-formatted String
    * @return config as InstanceOf[InferenceMainConfig]
    */
  def convertJsonConfigToClass(jsonConfig: String): InferenceMainConfig = {

    implicit val formats: Formats = Serialization.formats(hints = NoTypeHints)
    read[InferenceMainConfig](jsonConfig)

  }

  /**
    * Seems a bit counter-intuitive to do this, but this allows for cloud-agnostic storage of the config.
    * Otherwise, a configuration would need to be created to manage which cloud this is operating on and handle
    * native SDK object writers.  Instead of re-inventing the wheel here, a DataFrame can be serialized to
    * any cloud-native storage medium with very little issue.
    * @param config The inference configuration generated for a particular modeling run
    * @return A DataFrame consisting of a single row and a single field.  Cell 1:1 contains the json string.
    */
  def convertInferenceConfigToDataFrame(config: InferenceMainConfig): DataFrame = {

    import spark.sqlContext.implicits._

    val jsonConfig = convertInferenceConfigToJson(config)

    sc.parallelize(Seq(jsonConfig.compactJson)).toDF("config")

  }

  /**
    * From a supplied DataFrame that contains the configuration in cell 1:1, get the json string
    * @param configDataFrame A Dataframe that contains the configuration for the Inference run.
    * @return The string-encoded json payload for InferenceMainConfig
    */
  def extractInferenceJsonFromDataFrame(configDataFrame: DataFrame): String = {

    configDataFrame.collect()(0).get(0).toString

  }

  /**
    * Extract the InferenceMainConfig from a stored DataFrame containing the string-encoded json in row 1, column 1
    * @param configDataFrame A Dataframe that contains the configuration for the Inference run.
    * @return an instance of InferenceMainConfig
    */
  def extractInferenceConfigFromDataFrame(configDataFrame: DataFrame): InferenceMainConfig = {

    val encodedJson = extractInferenceJsonFromDataFrame(configDataFrame)

    convertJsonConfigToClass(encodedJson)

  }


}



