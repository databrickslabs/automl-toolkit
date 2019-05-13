package com.databricks.labs.automl.reports

import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

trait ReportingTools extends SparkSessionWrapper {

  def generateFrameReport(columns: Array[String], importances: Array[Double]): DataFrame = {
    import spark.sqlContext.implicits._
    sc.parallelize(columns zip importances).toDF("Feature", "Importance").orderBy($"Importance".desc)
      .withColumn("Importance", col("Importance") * 100.0)
      .withColumn("Feature", split(col("Feature"), "_si$")(0))
  }

  def cleanupFieldArray(indexedFields: Array[(String, Int)]): List[(String, Int)] = {

    val cleanedBuffer = new ListBuffer[(String, Int)]
    indexedFields.map(x => {
      cleanedBuffer += ((x._1.split("_si$")(0), x._2))
    })
    cleanedBuffer.result()
  }

  def generateDecisionTextReport(modelDebugString: String, featureIndex: List[(String, Int)]): String = {

    val reparsedArray = new ArrayBuffer[(String, String)]

    featureIndex.toArray.map(x => {
      reparsedArray += (("feature " + x._2.toString, x._1))
    })
    reparsedArray.result.toMap.foldLeft(modelDebugString){case(body, (k,v)) => body.replaceAll(k, v)}
  }

  def reportFields(fieldIndexArray: Array[(String, Int)]): String = {

    val stringConstructor = new ArrayBuffer[String]
    cleanupFieldArray(fieldIndexArray).foreach(x => {
      stringConstructor += s"Column ${x._1} is feature ${x._2}"
    })
    stringConstructor.result.mkString("\n")
  }

  def extractTopFeaturesByCount(featureFrame: DataFrame, topNCutoff: Int): Array[String] = {
    // Ensure the DataFrame is sorted and take the top N rows
    val sortedData = featureFrame.sort(col("Importance").desc).limit(topNCutoff).collect()

    sortedData.map(x => x(0).toString)

  }

  def extractTopFeaturesByImportance(featureFrame: DataFrame, importancePercentageCutoff: Double): Array[String] = {

    val sortedData = featureFrame.filter(col("Importance") >= importancePercentageCutoff)
      .sort(col("Importance").desc).collect()

    sortedData.map(x => x(0).toString)
  }


}
