package com.databricks.spark.automatedml.sanitize

import com.databricks.spark.automatedml.params.PearsonPayload
import com.databricks.spark.automatedml.utils.DataValidation
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

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


  private var _pearsonVectorFields: Array[String] = Array.empty
  private var _pearsonNonCategoricalFields: Array[String] = Array.empty

  private def setPearsonNonCategoricalFields(value: Array[String]): this.type = {
    _pearsonNonCategoricalFields = value
    this
  }

  private def setPearsonVectorFields(value: Array[String]): this.type = {
    _pearsonVectorFields = value
    this
  }

  /**
    * Private method for calculating the ChiSq relation of each feature to the label column.
    * @param data DataFrame that contains the vector to test and the label column.
    * @param featureColumn the name of the feature column vector to be used in the test.
    * @return List of the stats from the comparison calculated.
    */
  private def buildChiSq(data: DataFrame, featureColumn: String): List[PearsonPayload] = {
    val reportBuffer = new ListBuffer[PearsonPayload]

    val chi = ChiSquareTest.test(data, featureColumn, _labelCol).head
    val pvalues = chi.getAs[Vector](0).toArray
    val degreesFreedom = chi.getSeq[Int](1).toArray
    val pearsonStat = chi.getAs[Vector](2).toArray

    for(i <- _pearsonVectorFields.indices){
      reportBuffer += PearsonPayload(_pearsonVectorFields(i), pvalues(i), degreesFreedom(i), pearsonStat(i))
    }
    reportBuffer.result
  }

  /**
    * Method for, given a particular column, get the exact count of the cardinality of the field.
    * @param column Name of the column that is being tested for cardinality
    * @return [Long] the number of unique entries in the column
    */
  private def acquireCardinality(column: String): Long = {

    val aggregateData = df.select(col(column)).groupBy(col(column)).agg(count(col(column)))
    aggregateData.count()
  }

  /**
    * Private method for running through all of the fields included in the base feature vector and calculating their
    * cardinality in parallel (10x concurrency)
    * @return An Array of Field Name, Distinct Count
    */
  private def featuresCardinality(): Array[(String, Long)] = {

    val cardinalityOfFields = new ArrayBuffer[(String, Long)]()

    val featurePool = featureColumnListing.par
    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(10))
    featurePool.tasksupport = taskSupport

    featurePool.foreach{ x=>
      cardinalityOfFields += Tuple2(x, acquireCardinality(x))
    }

    cardinalityOfFields.result.toArray
  }

  /**
    * Private method for analyzing the input feature vector columns, determining their cardinality, and updating
    * the private var's to use these new lists.
    * @return Nothing - it updates the class-scoped variables when called.
    */
  private def restrictFeatureSet(): this.type = {

    // Empty ArrayBuffer to hold the fields to build the PearsonFeature Vector
    val pearsonVectorBuffer = new ArrayBuffer[String]
    val pearsonNonCategoricalBuffer = new ArrayBuffer[String]

    val determineCardinality = featuresCardinality()

    determineCardinality.foreach{ x=>
      if(x._2 < 10000) pearsonVectorBuffer += x._1 else pearsonNonCategoricalBuffer += x._1
    }

    setPearsonNonCategoricalFields(pearsonNonCategoricalBuffer.result.toArray)
    setPearsonVectorFields(pearsonVectorBuffer.result.toArray)

  }

  /**
    * Method for creating a new temporary feature vector that will be used for Pearson Filtering evaluation, removing
    * the high cardinality fields from this test.
    * @return [DataFrame] the DataFrame with a new vector entiitled "pearsonVector" that is used for removing
    *         fields from the feature vector that are either highly positively or negatively correlated to the label
    *         field.
    */
  private def reVectorize(): DataFrame = {

    // Create a new feature vector based on the fields that will be evaluated in PearsonFiltering
    restrictFeatureSet()

    require(_pearsonVectorFields.nonEmpty, s"Pearson Filtering contains all continuous variables in the feature" +
      s" vector, or cardinality of all features is greater than the threshold of 10k unique entries.  " +
      s"Please turn off pearson filtering for this data set by defining the main class with the setter: " +
      s".pearsonFilterOff() to continue.")

    val assembler = new VectorAssembler()
      .setInputCols(_pearsonVectorFields)
      .setOutputCol("pearsonVector")

    assembler.transform(df)
  }

  /**
    * Method for manually filtering out fields from the feature vector based on a user-supplied or
    * automation-calculated threshold cutoff.
    * @param statPayload the calculated correlation stats from feature elements in the vector to the label column.
    * @param filterValue the cut-off value specified by the user, or calculated through the quantile generator
    *                    methodology.
    * @return A list of fields that will be persisted and included in the feature vector going forward.
    */
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
      case "lesser" =>
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

  /**
    * Method for automatically detecting the quantile values for the filter statistic to cull fields automatically
    * based on the distribution of correlation amongst the feature vector and the label.
    * @param pearsonResults The pearson (and other) stats that have been calculated between each element of the
    *                       feature vector and the label.
    * @return The PearsonPayload results for each field, filtering out those elements that are either above / below
    *         the threshold configured.
    */
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

  /**
    * Main entry point for Pearson Filtering
    * @param ignoreFields Fields that will be ignored from running a Pearson filter against.
    * @return
    */
  def filterFields(ignoreFields: Array[String]=Array.empty[String]): DataFrame = {

    val revectoredData = reVectorize()

    val chiSqData = buildChiSq(revectoredData, "pearsonVector")
    val featureFields: List[String] = _filterMode match {
      case "manual" =>
        filterChiSq(chiSqData, _filterManualValue)
      case _ =>
        filterChiSq(chiSqData, quantileGenerator(chiSqData))
      }
    require(featureFields.nonEmpty, "All feature fields have been filtered out.  Adjust parameters.")
    val fieldListing = featureFields ::: List(_labelCol) ::: ignoreFields.toList ::: _pearsonNonCategoricalFields.toList
    df.select(fieldListing.map(col):_*)
  }

}
