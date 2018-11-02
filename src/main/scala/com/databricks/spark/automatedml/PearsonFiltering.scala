package com.databricks.spark.automatedml

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.{Matrix, Vectors, Vector}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
import org.apache.spark.ml.stat.ChiSquareTest
import scala.collection.mutable.ListBuffer

//TODO: finish this for dynamic and manual filtering of feature fields based on pearson relevance scores.


class PearsonFiltering(df: DataFrame) extends DataValidation {

  private var _labelCol: String = "label"
  private var _featuresCol: String = "features"
  private var _featureColumnsListing: Array[String] = Array.empty[String]
  private var _filterStatistic: String = "pearsonStat"
  private var _filterDirection: String = "greater"

  private var _filterManualValue: Double = 0.0
  private var _filterMode: String = "auto"


  final private val _dataFieldNames = df.schema.fieldNames
  final private val _allowedStats: Array[String] = Array("pvalue", "degreesFreedom", "pearsonStat")
  final private val _allowedFilterDirections: Array[String] = Array("greater", "lesser")
  final private val _allowedFilterModes: Array[String] = Array("auto", "manual")

  def setLabelCol(value: String): this.type = {
    assert(_dataFieldNames.contains(value), s"Label Field $value is not in DataFrame Schema.")
    _labelCol = value
    this
  }

  def setFeaturesCol(value: String): this.type = {
    assert(_dataFieldNames.contains(value), s"Feature Field $value is not in DataFrame Schema.")
    _featuresCol = value
    this
  }

  def setFeatureColumnsListing(value: Array[String]): this.type = {
    assert(value.length > 0, s"Size of feature colums listing array must not be empty.")
    _featureColumnsListing = value
    this
  }

  def setFilterStatistic(value: String): this.type = {
    assert(_allowedStats.contains(value), s"Pearson Filtering Statistic '$value' is not a valid member of ${
      invalidateSelection(value, _allowedStats)}")
    _filterStatistic = value
    this
  }

  def setFilterDirection(value: String): this.type = {
    assert(_allowedFilterDirections.contains(value), s"Filter Direction '$value' is not a valid member of ${
      invalidateSelection(value, _allowedFilterDirections)
    }")
    _filterDirection = value
    this
  }

  def setFilterManualValue(value: Double): this.type = {
    _filterManualValue = value
    this
  }

  def setFilterMode(value: String): this.type = {
    assert(_allowedFilterModes.contains(value), s"Filter Mode $value is not a valid member of ${
      invalidateSelection(value, _allowedFilterModes)}")
    _filterMode = value
    this
  }

  //TODO: getters setters for filtering mode / manual value, provide quantile representation for automatic mode.

  def getLabelCol: String = _labelCol
  def getFeaturesCol: String = _featuresCol
  def getFeatureColumnsListing: Array[String] = _featureColumnsListing
  def getFilterStatistic: String = _filterStatistic
  def getFilterDirection: String = _filterDirection
  def getFilterManualValue: Double = _filterManualValue
  def getFilterMode: String = _filterMode


  private def buildChiSq(): List[PearsonPayload] = {
    val reportBuffer = new ListBuffer[PearsonPayload]

    val chi = ChiSquareTest.test(df, "features", "label").head
    val pvalues = chi.getAs[Vector](0).toArray
    val degreesFreedom = chi.getSeq[Int](1).toArray
    val pearsonStat = chi.getAs[Vector](2).toArray

    for(i <- _featureColumnsListing.indices){
      reportBuffer += PearsonPayload(_featureColumnsListing(i), pvalues(i), degreesFreedom(i), pearsonStat(i))
    }
    reportBuffer.result
  }

  def manualFilterChiSq(statPayload: List[PearsonPayload], filterValue: Double): List[String] = {
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

  def quantileGenerator(percentile: Double, stat: String, pearsonResults: List[PearsonPayload]): Double = {

    assert(percentile < 1 & percentile > 0, "Percentile Value must be between 0 and 1.")
    val statBuffer = new ListBuffer[Double]
    pearsonResults.foreach(x => {
      x.getClass.getDeclaredFields foreach {f=>
        f.setAccessible(true)
        if(f.getName == stat) statBuffer += f.get(x).asInstanceOf[Double]
      }
    })

    val statSorted = statBuffer.result.sortWith(_<_)
    if(statSorted.size % 2 == 1) statSorted((statSorted.size * percentile).toInt)
    else {
      val splitLoc = math.floor(statSorted.size * percentile).toInt
      val splitCheck = if(splitLoc < 1) 1 else splitLoc.toInt
      val(high, low) = statSorted.splitAt(splitCheck)
      (high.last + low.head) / 2
    }

  }

  //TODO: public accessor main method for doing either manual or automated filtering of the pearson data - returning a DataFrame.

}
