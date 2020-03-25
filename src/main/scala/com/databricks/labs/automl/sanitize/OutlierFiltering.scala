package com.databricks.labs.automl.sanitize

import java.util.regex.Pattern

import com.databricks.labs.automl.exceptions.ThreadPoolsBySize
import com.databricks.labs.automl.params.{FilterData, ManualFilters}
import com.databricks.labs.automl.utils.{
  DataValidation,
  SchemaUtils,
  SparkSessionWrapper
}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame}

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool
import scala.concurrent.{Await, Future}

/**
  *
  * @param df - Input DataFrame pre-feature vectorization
  */
class OutlierFiltering(df: DataFrame)
    extends SparkSessionWrapper
    with DataValidation {

  private case class OutlierFilteredDf(mutatedDf: DataFrame,
                                       outlierDf: DataFrame)

  private lazy val LOWER = "lower"
  private lazy val UPPER = "upper"
  private lazy val BOTH = "both"

  private var _labelCol: String = "label"
  private var _filterBounds: String = BOTH
  private var _lowerFilterNTile: Double = 0.02
  private var _upperFilterNTile: Double = 0.98
  private var _filterPrecision: Double = 0.01
  private var _continuousDataThreshold: Int = 50
  private var _parallelism: Int = 20

  final private val _filterBoundaryAllowances: Array[String] =
    Array(LOWER, UPPER, BOTH)
  final private val _dfSchema = df.schema.fieldNames

  def setLabelCol(value: String): this.type = {
    require(
      _dfSchema.contains(value),
      s"DataFrame does not contain label column $value"
    )
    this._labelCol = value
    this
  }

  def setFilterBounds(value: String): this.type = {
    require(
      _filterBoundaryAllowances.contains(value),
      s"Filter Boundary Mode $value is not a valid member of " +
        s"${invalidateSelection(value, _filterBoundaryAllowances)}"
    )
    this._filterBounds = value
    this
  }

  def setLowerFilterNTile(value: Double): this.type = {
    require(
      value >= 0.0 & value <= 1.0,
      s"Lower Filter NTile must be between 0.0 and 1.0"
    )
    this._lowerFilterNTile = value
    this
  }

  def setUpperFilterNTile(value: Double): this.type = {
    require(
      value >= 0.0 & value <= 1.0,
      s"Upper Filter NTile must be between 0.0 and 1.0"
    )
    this._upperFilterNTile = value
    this
  }

  def setFilterPrecision(value: Double): this.type = {
    if (value == 0.0)
      println(
        "Warning! Precision of 0 is an exact calculation of quantiles and may not be performant!"
      )
    this._filterPrecision = value
    this
  }

  def setContinuousDataThreshold(value: Int): this.type = {
    if (value < 50)
      println("Warning! Values less than 50 may indicate oridinal data!")
    this._continuousDataThreshold = value
    this
  }

  def setParallelism(value: Int): this.type = {
    _parallelism = value
    this
  }

  def getLabelCol: String = _labelCol
  def getFilterBounds: String = _filterBounds
  def getLowerFilterNTile: Double = _lowerFilterNTile
  def getUpperFilterNTile: Double = _upperFilterNTile
  def getFilterPrecision: Double = _filterPrecision
  def getContinuousDataThreshold: Int = _continuousDataThreshold
  def getParallelism: Int = _parallelism

  private def filterBoundaries(field: String, ntile: Double): Double = {
    df.stat.approxQuantile(field, Array(ntile), _filterPrecision)(0)
  }

  private def getBatches(items: List[String]): Array[List[String]] = {
    val batches = ArrayBuffer[List[String]]()
    val batchSize = items.length / _parallelism
    for (i <- 0 to items.length by batchSize) {
      batches.append(items.slice(i, i + batchSize))
    }
    batches.toArray
  }

  private def validateNumericFields(
    ignoreList: Array[String]
  ): (List[FilterData], List[String]) = {

    val numericFieldReport = new ListBuffer[FilterData]
    val fields = SchemaUtils.extractTypes(df, _labelCol, ignoreList)
    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(_parallelism))
    val numericFieldBatches = getBatches(fields.numericFields).par
    numericFieldBatches.tasksupport = taskSupport

    numericFieldBatches.foreach { batch =>
      val countFields = ArrayBuffer[Column]()
      batch.foreach(batchCol => {
        countFields.append(approx_count_distinct(batchCol, _filterPrecision))
      })
      val countsByCol = batch zip df
        .select(countFields: _*)
        .collect()(0)
        .toSeq
        .toArray
        .map(_.asInstanceOf[Long])
      if (countsByCol.nonEmpty)
        numericFieldReport += FilterData(
          countsByCol.head._1,
          countsByCol.head._2
        )
    }
    val totalFields = fields.numericFields ::: fields.categoricalFields
    (numericFieldReport.result(), totalFields)
  }

  private def filterLow(data: DataFrame,
                        field: String,
                        filterThreshold: Double): OutlierFilteredDf = {
    OutlierFilteredDf(
      data.filter(col(field) >= filterThreshold),
      data.filter(col(field) < filterThreshold)
    )
  }

  private def filterHigh(data: DataFrame,
                         field: String,
                         filterThreshold: Double): OutlierFilteredDf = {
    OutlierFilteredDf(
      data.filter(col(field) <= filterThreshold),
      data.filter(col(field) > filterThreshold)
    )
  }

  def filterContinuousOutliers(
    vectorIgnoreList: Array[String],
    ignoreList: Array[String] = Array.empty[String]
  ): (DataFrame, DataFrame, Map[String, (Double, String)]) = {
    val filteredNumericPayload = new ListBuffer[String]
    val (numericPayload, totalFeatureFields) = validateNumericFields(
      vectorIgnoreList ++ ignoreList
    )

    val totalFields = totalFeatureFields ++ List(_labelCol) ++ vectorIgnoreList.toList ++ ignoreList.toList
    numericPayload.foreach { x =>
      if (!ignoreList.contains(x.field) & x.uniqueValues >= _continuousDataThreshold)
        filteredNumericPayload += x.field
    }
    var mutatedDF = df
    var outlierDF = df
    val inferenceOutlierMap =
      addToInferenceOutlierMap(filteredNumericPayload.toList, _filterBounds)
    inferenceOutlierMap.foreach(item => {
      val colName = item._1.split(Pattern.quote("||"))(0)
      item._2._2 match {
        case LOWER =>
          val outlierDfs = filterLow(mutatedDF, colName, item._2._1)
          mutatedDF = outlierDfs.mutatedDf
          outlierDF = outlierDfs.outlierDf
        case UPPER =>
          val outlierDfs = filterHigh(mutatedDF, colName, item._2._1)
          mutatedDF = outlierDfs.mutatedDf
          outlierDF = if (BOTH.equals(_filterBounds)) {
            outlierDfs.outlierDf.union(outlierDF)
          } else {
            outlierDfs.outlierDf
          }
      }
    })
    (
      mutatedDF.select(totalFields.distinct map col: _*),
      outlierDF.select(totalFields.distinct map col: _*),
      inferenceOutlierMap.result().toMap
    )
  }

  private def addToInferenceOutlierMap(
    filteredData: List[String],
    filterDirection: String
  ): mutable.Map[String, (Double, String)] = {
    val executionContext =
      ThreadPoolsBySize.withScalaExecutionContext(_parallelism)
    var outlierMap = mutable.Map[String, (Double, String)]()
    val colFutures = ArrayBuffer[Future[(String, (Double, String))]]()
    filteredData.foreach(colName => {
      getFilterNTileByCase(filterDirection)
        .foreach(
          item =>
            colFutures += Future {
              colName + "||" + item._1 -> (filterBoundaries(colName, item._2), item._1)
            }(executionContext)
        )
    })
    colFutures.foreach(item => {
      val outlier = Await.result(item, scala.concurrent.duration.Duration.Inf)
      outlierMap += outlier
    })
    outlierMap
  }

  private def getFilterNTileByCase(
    filterDirection: String
  ): Array[(String, Double)] = {
    filterDirection match {
      case LOWER => Array((LOWER, _lowerFilterNTile))
      case UPPER => Array((UPPER, _upperFilterNTile))
      case BOTH  => Array((LOWER, _lowerFilterNTile), (UPPER, _upperFilterNTile))
    }
  }

  def filterContinuousOutliers(
    manualFilter: List[ManualFilters],
    vectorIgnoreList: Array[String]
  ): (DataFrame, DataFrame, Map[String, (Double, String)]) = {

    var mutatedDF = df
    var outlierDF = df
    val (numericPayload, totalFeatureFields) = validateNumericFields(
      vectorIgnoreList
    )
    val totalFields = totalFeatureFields ++ List(_labelCol) ++ vectorIgnoreList.toList

    val inferenceOutlierMap: mutable.Map[String, (Double, String)] =
      mutable.Map.empty[String, (Double, String)]

    manualFilter.foreach { x =>
      _filterBounds match {
        case LOWER =>
          inferenceOutlierMap.put(x.field, (x.threshold, LOWER))
          val outlierDfs = filterLow(mutatedDF, x.field, x.threshold)
          mutatedDF = outlierDfs.mutatedDf
          outlierDF = outlierDfs.outlierDf
        case UPPER =>
          inferenceOutlierMap.put(x.field, (x.threshold, UPPER))
          val outlierDfs = filterHigh(mutatedDF, x.field, x.threshold)
          mutatedDF = outlierDfs.mutatedDf
          outlierDF = if (BOTH.equals(_filterBounds)) {
            outlierDfs.outlierDf.union(outlierDF)
          } else {
            outlierDfs.outlierDf
          }
        case _ =>
          throw new UnsupportedOperationException(
            s"Filter mode '${_filterBounds} is not supported.  Please use either '$LOWER' or '$UPPER'"
          )
      }
    }
    (
      mutatedDF.select(totalFields.distinct map col: _*),
      outlierDF.select(totalFields.distinct map col: _*),
      inferenceOutlierMap.result.toMap
    )
  }
}
