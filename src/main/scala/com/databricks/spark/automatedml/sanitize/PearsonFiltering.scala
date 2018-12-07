package com.databricks.spark.automatedml.sanitize

import com.databricks.spark.automatedml.params.PearsonPayload
import com.databricks.spark.automatedml.utils.DataValidation
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ListBuffer

/**
  *
  * @param df                   : DataFrame -> Dataset with a vectorized field of features,
  *                             the feature columns, and a label column.
  * @param featureColumnListing : Array[String] -> List of all fields that make up the feature vector
  *
  *                             Usage:
  *                             val autoFiltered = new PearsonFiltering(featurizedData, fields)
  *                             .setLabelCol("label")
  *                             .setFeaturesCol("features")
  *                             .setFilterStatistic("pearsonStat")
  *                             .setFilterDirection("greater")
  *                             .setFilterMode("auto")
  *                             .setAutoFilterNTile(0.5)
  *                             .filterFields
  */

class PearsonFiltering(df: DataFrame, featureColumnListing: Array[String]) extends DataValidation
  with SanitizerDefaults {

  private var _labelCol: String = defaultLabelCol
  private var _featuresCol: String = defaultFeaturesCol
  private var _filterStatistic: String = defaultPearsonFilterStatistic
  private var _filterDirection: String = defaultPearsonFilterDirection

  private var _filterManualValue: Double = defaultPearsonFilterManualValue
  private var _filterMode: String = defaultPearsonFilterMode
  private var _autoFilterNTile: Double = defaultPearsonAutoFilterNTile


  final private val _dataFieldNames = df.schema.fieldNames
  final private val _dataFieldTypes = df.schema.fields


  def setLabelCol(value: String): this.type = {
    require(_dataFieldNames.contains(value), s"Label Field $value is not in DataFrame Schema.")
    _labelCol = value
    this
  }

  def setFeaturesCol(value: String): this.type = {
    require(_dataFieldNames.contains(value), s"Feature Field $value is not in DataFrame Schema.")
    require(_dataFieldTypes.filter(_.name == value)(0).dataType.typeName == "vector",
      s"Feature Field $value is not of vector type.")
    _featuresCol = value
    this
  }

  def setFilterStatistic(value: String): this.type = {
    require(_allowedStats.contains(value), s"Pearson Filtering Statistic '$value' is not a valid member of ${
      invalidateSelection(value, _allowedStats)}")
    _filterStatistic = value
    this
  }

  def setFilterDirection(value: String): this.type = {
    require(_allowedFilterDirections.contains(value), s"Filter Direction '$value' is not a valid member of ${
      invalidateSelection(value, _allowedFilterDirections)
    }")
    _filterDirection = value
    this
  }

  def setFilterManualValue(value: Double): this.type = {
    _filterManualValue = value
    this
  }

  def setFilterManualValue(value: Int): this.type = {
    _filterManualValue = value.toDouble
    this
  }

  def setFilterMode(value: String): this.type = {
    require(_allowedFilterModes.contains(value), s"Filter Mode $value is not a valid member of ${
      invalidateSelection(value, _allowedFilterModes)}")
    _filterMode = value
    this
  }

  def setAutoFilterNTile(value: Double): this.type = {
    require(value <= 1.0 & value >= 0.0, "NTile value must be between 0 and 1.")
    _autoFilterNTile = value
    this
  }

  def getLabelCol: String = _labelCol
  def getFeaturesCol: String = _featuresCol
  def getFilterStatistic: String = _filterStatistic
  def getFilterDirection: String = _filterDirection
  def getFilterManualValue: Double = _filterManualValue
  def getFilterMode: String = _filterMode
  def getAutoFilterNTile: Double = _autoFilterNTile

  private def buildChiSq(): List[PearsonPayload] = {
    val reportBuffer = new ListBuffer[PearsonPayload]

    val chi = ChiSquareTest.test(df, _featuresCol, _labelCol).head
    val pvalues = chi.getAs[Vector](0).toArray
    val degreesFreedom = chi.getSeq[Int](1).toArray
    val pearsonStat = chi.getAs[Vector](2).toArray

    for(i <- featureColumnListing.indices){
      reportBuffer += PearsonPayload(featureColumnListing(i), pvalues(i), degreesFreedom(i), pearsonStat(i))
    }
    reportBuffer.result
  }

  private def filterChiSq(statPayload: List[PearsonPayload], filterValue: Double): List[String] = {
    val fieldRestriction = new ListBuffer[String]
    _filterDirection match {
      case "greater" =>
        statPayload.foreach(x => {
          x.getClass.getDeclaredFields foreach {f =>
            f.setAccessible(true)
            if(f.getName == _filterStatistic)
              if(f.get(x).asInstanceOf[Double] >= filterValue)
                fieldRestriction += x.fieldName
              else None
            else None
          }
        })
      case "less" =>
        statPayload.foreach(x => {
          x.getClass.getDeclaredFields foreach {f =>
            f.setAccessible(true)
            if(f.getName == _filterStatistic)
              if(f.get(x).asInstanceOf[Double] <= filterValue)
                fieldRestriction += x.fieldName
              else None
            else None
          }
        })
      case _ => throw new UnsupportedOperationException(s"${_filterDirection} is not supported for manualFilterChiSq")    }
    fieldRestriction.result
  }

  private def quantileGenerator(pearsonResults: List[PearsonPayload]): Double = {

    val statBuffer = new ListBuffer[Double]
    pearsonResults.foreach(x => {
      x.getClass.getDeclaredFields foreach {f=>
        f.setAccessible(true)
        if(f.getName == _filterStatistic) statBuffer += f.get(x).asInstanceOf[Double]
      }
    })

    val statSorted = statBuffer.result.sortWith(_<_)
    if(statSorted.size % 2 == 1) statSorted((statSorted.size * _autoFilterNTile).toInt)
    else {
      val splitLoc = math.floor(statSorted.size * _autoFilterNTile).toInt
      val splitCheck = if(splitLoc < 1) 1 else splitLoc.toInt
      val(high, low) = statSorted.splitAt(splitCheck)
      (high.last + low.head) / 2
    }

  }

  def filterFields(): DataFrame = {

    val chiSqData = buildChiSq()
    val featureFields: List[String] = _filterMode match {
      case "manual" =>
        filterChiSq(chiSqData, _filterManualValue)
      case _ =>
        filterChiSq(chiSqData, quantileGenerator(chiSqData))
      }
    require(featureFields.nonEmpty, "All feature fields have been filtered out.  Adjust parameters.")
    val fieldListing = featureFields ::: List(_labelCol)
    df.select(fieldListing.map(col):_*)
  }

}
