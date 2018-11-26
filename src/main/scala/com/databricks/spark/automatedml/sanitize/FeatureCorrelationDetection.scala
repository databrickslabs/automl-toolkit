package com.databricks.spark.automatedml.sanitize

import com.databricks.spark.automatedml.params.FeatureCorrelationStats
import com.databricks.spark.automatedml.utils.SparkSessionWrapper
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer

class FeatureCorrelationDetection(data: DataFrame, fieldListing: Array[String]) extends SparkSessionWrapper {

  private var _correlationCutoffHigh: Double = 0.0
  private var _correlationCutoffLow: Double = 0.0
  private var _labelCol: String = "label"

  final private val _dataFieldNames = data.schema.fieldNames

  def setCorrelationCutoffHigh(value: Double): this.type = {
    require(value < 1.0, "Maximum range of Correlation Cutoff on the high end must be less than 1.0")
    _correlationCutoffHigh = value
    this
  }

  def setCorrelationCutoffLow(value: Double): this.type = {
    require(value > -1.0, "Minimum range of Correlation Cutoff on the low end must be greater than -1.0")
    _correlationCutoffLow = value
    this
  }

  def setLabelCol(value: String): this.type = {
    require(_dataFieldNames.contains(value), s"Label field $value is not in Dataframe")
    _labelCol = value
    this
  }

  def getCorrelationCutoffHigh: Double = _correlationCutoffHigh

  def getCorrelationCutoffLow: Double = _correlationCutoffLow

  def getLabelCol: String = _labelCol

  private def computeFeatureCorrelation(): Array[FeatureCorrelationStats] = {

    val correlationInteractions = new ArrayBuffer[FeatureCorrelationStats]
    val redundantRecursionEliminator = new ArrayBuffer[String]

    fieldListing.foreach{ x =>
      val leftFields = fieldListing.filterNot(_.contains(x)).filterNot(f => redundantRecursionEliminator.contains(f))
      leftFields.foreach{y =>
        correlationInteractions += FeatureCorrelationStats(x, y, data.groupBy().agg(corr(x, y).as("pearson"))
          .first().getDouble(0))
      }
      redundantRecursionEliminator += x
    }

    correlationInteractions.result.toArray
  }

  private def generateFilterFields(): Seq[String] = {

    val featureCorrelationData = computeFeatureCorrelation()

    featureCorrelationData.filter(x => x.correlation > _correlationCutoffHigh || x.correlation < _correlationCutoffLow)
      .map(_.rightCol).toSeq

  }

  // Manual debugging mode public method
  def generateFeatureCorrelationReport(): DataFrame = {

    import spark.sqlContext.implicits._

    sc.parallelize(computeFeatureCorrelation()).toDF
  }

  def filterFeatureCorrelation(): DataFrame = {

    assert(_dataFieldNames.contains(_labelCol), s"Label field ${_labelCol} is not in Dataframe")
    val fieldsToFilter = generateFilterFields()
    assert(fieldListing.length > fieldsToFilter.length,
      s"All feature fields have been removed and modeling cannot continue.")
    val fieldsToSelect = _dataFieldNames.filterNot(f => fieldsToFilter.contains(f))
    data.select(fieldsToSelect map col:_*)
  }

}
