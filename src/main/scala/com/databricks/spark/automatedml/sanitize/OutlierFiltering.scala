package com.databricks.spark.automatedml.sanitize

import com.databricks.spark.automatedml.params.{FilterData, ManualFilters}
import com.databricks.spark.automatedml.utils.{DataValidation, SparkSessionWrapper}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ListBuffer

/**
  *
  * @param df - Input DataFrame pre-feature vectorization
  */

class OutlierFiltering(df: DataFrame) extends SparkSessionWrapper with DataValidation{

  private var _labelCol: String = "label"
  private var _filterBounds: String = "both"
  private var _lowerFilterNTile: Double = 0.02
  private var _upperFilterNTile: Double = 0.98
  private var _filterPrecision: Double = 0.01
  private var _continuousDataThreshold: Int = 50

  final private val _filterBoundaryAllowances: Array[String] = Array("lower", "upper", "both")
  final private val _dfSchema = df.schema.fieldNames

  def setLabelCol(value: String): this.type = {
    require(_dfSchema.contains(value), s"DataFrame does not contain label column $value")
    this._labelCol = value
    this
  }

  def setFilterBounds(value: String): this.type = {
    require(_filterBoundaryAllowances.contains(value),
      s"Filter Boundary Mode $value is not a valid member of " +
        s"${invalidateSelection(value, _filterBoundaryAllowances)}")
    this._filterBounds = value
    this
  }

  def setLowerFilterNTile(value: Double): this.type = {
    require(value >= 0.0 & value <= 1.0, s"Lower Filter NTile must be between 0.0 and 1.0")
    this._lowerFilterNTile = value
    this
  }

  def setUpperFilterNTile(value: Double): this.type = {
    require(value >= 0.0 & value <= 1.0, s"Upper Filter NTile must be between 0.0 and 1.0")
    this._upperFilterNTile = value
    this
  }

  def setFilterPrecision(value: Double): this.type = {
    if(value == 0.0) println("Warning! Precision of 0 is an exact calculation of quantiles and may not be performant!")
    this._filterPrecision = value
    this
  }

  def setContinuousDataThreshold(value: Int): this.type = {
    if(value < 50) println("Warning! Values less than 50 may indicate oridinal data!")
    this._continuousDataThreshold = value
    this
  }

  def getLabelCol: String = _labelCol
  def getFilterBounds: String = _filterBounds
  def getLowerFilterNTile: Double = _lowerFilterNTile
  def getUpperFilterNTile: Double = _upperFilterNTile
  def getFilterPrecision: Double = _filterPrecision
  def getContinuousDataThreshold: Int = _continuousDataThreshold

  private def filterBoundaries(field: String, ntile: Double): Double = {
    df.stat.approxQuantile(field, Array(ntile), _filterPrecision)(0)
  }

  private def numericUniqueness(field: String): Long = {
    df.select(field).distinct().count()
  }

  private def validateNumericFields(ignoreList: Array[String]): (List[FilterData], List[String]) = {

    val numericFieldReport = new ListBuffer[FilterData]
    val (numericFields, characterFields, dateFields, timeFields) = extractTypes(df, _labelCol, ignoreList)
    numericFields.foreach{x =>
      numericFieldReport += FilterData(x, numericUniqueness(x))
    }
    val totalFields = numericFields ::: characterFields
    (numericFieldReport.result(), totalFields)
  }

  private def filterLow(data: DataFrame, field: String, filterThreshold: Double): DataFrame = {
    data.filter(col(field) >= filterThreshold)
  }

  private def filterHigh(data: DataFrame, field: String, filterThreshold: Double): DataFrame = {
    data.filter(col(field) <= filterThreshold)
  }

  def filterContinuousOutliers(vectorIgnoreList: Array[String], ignoreList: Array[String]=Array.empty[String]):
  (DataFrame, DataFrame) = {

    val filteredNumericPayload = new ListBuffer[String]
    val (numericPayload, totalFeatureFields) = validateNumericFields(vectorIgnoreList)
    val totalFields = totalFeatureFields ++ List(_labelCol) ++ vectorIgnoreList.toList
    numericPayload.foreach{x =>
      if(!ignoreList.contains(x.field) & x.uniqueValues >= _continuousDataThreshold)
        filteredNumericPayload += x.field
    }
    var mutatedDF = df
    var outlierDF = df
    filteredNumericPayload.foreach{x =>
      _filterBounds match {
        case "lower" =>
          mutatedDF = filterLow(mutatedDF, x, filterBoundaries(x, _lowerFilterNTile))
          outlierDF = filterHigh(outlierDF, x, filterBoundaries(x, _lowerFilterNTile))
        case "upper" =>
          mutatedDF = filterHigh(mutatedDF, x, filterBoundaries(x, _upperFilterNTile))
          outlierDF = filterLow(outlierDF, x, filterBoundaries(x, _upperFilterNTile))
        case "both" =>
          mutatedDF = filterLow(mutatedDF, x, filterBoundaries(x, _lowerFilterNTile))
          mutatedDF = filterHigh(mutatedDF, x, filterBoundaries(x, _upperFilterNTile))
          outlierDF = filterHigh(outlierDF, x, filterBoundaries(x, _lowerFilterNTile))
          outlierDF = filterLow(outlierDF, x, filterBoundaries(x, _upperFilterNTile))
      }
    }
    (mutatedDF.select(totalFields.distinct map col: _*), outlierDF.select(totalFields.distinct map col: _*))
  }

  def filterContinuousOutliers(manualFilter: List[ManualFilters], vectorIgnoreList: Array[String]):
  (DataFrame, DataFrame) = {

    var mutatedDF = df
    var outlierDF = df
    val (numericPayload, totalFeatureFields) = validateNumericFields(vectorIgnoreList)
    val totalFields = totalFeatureFields ++ List(_labelCol) ++ vectorIgnoreList.toList
    manualFilter.foreach{x =>
      _filterBounds match {
        case "lower" =>
          mutatedDF = filterLow(mutatedDF, x.field, x.threshold)
          outlierDF = filterHigh(outlierDF, x.field, x.threshold)
        case "upper" =>
          mutatedDF = filterHigh(mutatedDF, x.field, x.threshold)
          outlierDF = filterLow(outlierDF, x.field, x.threshold)
        case _ => throw new UnsupportedOperationException(
          s"Filter mode '${_filterBounds} is not supported.  Please use either 'lower' or 'upper'")
      }
    }
    (mutatedDF.select(totalFields.distinct map col: _*), outlierDF.select(totalFields.distinct map col: _*))
  }
}
