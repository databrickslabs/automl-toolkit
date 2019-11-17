package com.databricks.labs.automl.sanitize

import com.databricks.labs.automl.params.FeatureCorrelationStats
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

class FeatureCorrelationDetection(data: DataFrame, fieldListing: Array[String]) extends SparkSessionWrapper {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _correlationCutoffHigh: Double = 0.0
  private var _correlationCutoffLow: Double = 0.0
  private var _labelCol: String = "label"
  private var _parallelism: Int = 20

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

  def setParallelism(value: Int): this.type = {
    _parallelism = value
    this
  }

  def getCorrelationCutoffHigh: Double = _correlationCutoffHigh

  def getCorrelationCutoffLow: Double = _correlationCutoffLow

  def getLabelCol: String = _labelCol
  def getParallelism: Int = _parallelism

  private def computeFeatureCorrelation(): Array[FeatureCorrelationStats] = {

    val correlationInteractions = new ArrayBuffer[FeatureCorrelationStats]
    val redundantRecursionEliminator = new ArrayBuffer[String]

    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(_parallelism))
    val fieldListingPar = fieldListing.par
    fieldListingPar.tasksupport = taskSupport


    //TODO (Jas) Tech Debt: Need to fold this in either direction
    fieldListingPar.foreach{ x =>
      val leftFields = fieldListing.filterNot(_.contains(x)).filterNot(f => redundantRecursionEliminator.contains(f))
      leftFields.foreach{y =>
        val corrStats: Double = try {
          data.groupBy().agg(corr(x, y).as("pearson")).first().getDouble(0)
        } catch {
          case e: java.lang.NullPointerException =>
            val errorMsg = s"Correlation Calculation for $x : $y failed.  Recording Inf for correlation."
            println(errorMsg)
            logger.log(Level.INFO, errorMsg + s"\n ${e.printStackTrace()}")
            Double.PositiveInfinity
        }
        correlationInteractions += FeatureCorrelationStats(x, y, corrStats)
      }
      redundantRecursionEliminator += x
    }

    correlationInteractions.result.toArray
  }

  private def generateFilterFields(): Seq[String] = {

    val featureCorrelationData = computeFeatureCorrelation()

    featureCorrelationData.filter(x => x.correlation < _correlationCutoffHigh && x.correlation > _correlationCutoffLow)
      .map(_.rightCol).toSeq

  }

  // Manual debugging mode public method
  def generateFeatureCorrelationReport(): DataFrame = {

    import spark.sqlContext.implicits._

    sc.parallelize(computeFeatureCorrelation()).toDF
  }

  def filterFeatureCorrelation(): DataFrame = {

    assert(_dataFieldNames.contains(_labelCol), s"Label field ${_labelCol} is not in Dataframe")
    val fieldsToKeep = generateFilterFields().distinct
    val fieldsToDrop = fieldListing.filterNot(f => fieldsToKeep.contains(f))
    assert(fieldListing.distinct.length > fieldsToDrop.length,
      s"All feature fields have been removed and modeling cannot continue.")
    data.drop(fieldsToDrop: _*)
  }

}
