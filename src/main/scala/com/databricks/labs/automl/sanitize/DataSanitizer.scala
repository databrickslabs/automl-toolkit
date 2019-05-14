package com.databricks.labs.automl.sanitize

import com.databricks.labs.automl.inference.NaFillConfig
import com.databricks.labs.automl.utils.DataValidation
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

class DataSanitizer(data: DataFrame) extends DataValidation {

  private var _labelCol = "label"
  private var _featureCol = "features"
  private var _numericFillStat = "mean"
  private var _characterFillStat = "max"
  private var _modelSelectionDistinctThreshold = 10
  private var _fieldsToIgnoreInVector = Array.empty[String]
  private var _filterPrecision: Double = 0.01
  private var _parallelism: Int = 20

  def setLabelCol(value: String): this.type = {
    this._labelCol = value
    this
  }

  def setFeatureCol(value: String): this.type = {
    this._featureCol = value
    this
  }

  def setNumericFillStat(value: String): this.type = {
    this._numericFillStat = value
    this
  }

  def setCharacterFillStat(value: String): this.type = {
    this._characterFillStat = value
    this
  }

  def setModelSelectionDistinctThreshold(value: Int): this.type = {
    this._modelSelectionDistinctThreshold = value
    this
  }

  def setFieldsToIgnoreInVector(value: Array[String]): this.type = {
    _fieldsToIgnoreInVector = value
    this
  }

  def setParallelism(value: Int): this.type = {
    _parallelism = value
    this
  }

  def setFilterPrecision(value: Double): this.type = {
    if (value == 0.0) println("Warning! Precision of 0 is an exact calculation of quantiles and may not be performant!")
    this._filterPrecision = value
    this
  }

  def getLabel: String = _labelCol

  def getFeatureCol: String = _featureCol

  def getNumericFillStat: String = _numericFillStat

  def getCharacterFillStat: String = _characterFillStat

  def getModelSelectionDistinctThreshold: Int = _modelSelectionDistinctThreshold

  def getFieldsToIgnoreInVector: Array[String] = _fieldsToIgnoreInVector

  def getParallelism: Int = _parallelism

  def getFilterPrecision: Double = _filterPrecision


  private var _labelValidation: Boolean = false

  private def labelValidationOn(): Boolean = true


  private def convertLabel(df: DataFrame): DataFrame = {

    val stringIndexer = new StringIndexer()
      .setInputCol(this._labelCol)
      .setOutputCol(this._labelCol + "_si")

    stringIndexer.fit(data).transform(data)
      .withColumn(this._labelCol, col(s"${this._labelCol}_si"))
      .drop(this._labelCol + "_si")
  }

  private def refactorLabel(df: DataFrame, labelColumn: String): DataFrame = {

    extractSchema(df.schema).foreach(x =>
      x._2 match {
        case `labelColumn` => x._1 match {
          case StringType => labelValidationOn()
          case BooleanType => labelValidationOn()
          case BinaryType => labelValidationOn()
          case _ => None
        }
        case _ => None
      })
    if (_labelValidation) convertLabel(df) else df
  }

  private def metricConversion(metric: String): String = {

    val allowableFillArray = Array("min", "25p", "mean", "median", "75p", "max")

    assert(allowableFillArray.contains(metric),
      s"The metric supplied, '$metric' is not in: " +
        s"${invalidateSelection(metric, allowableFillArray)}")

    val summaryMetric = metric match {
      case "25p" => "25%"
      case "median" => "50%"
      case "75p" => "75%"
      case _ => metric
    }
    summaryMetric
  }

  private def getBatches(items: List[String]): Array[List[String]] = {
    val batches = ArrayBuffer[List[String]]()
    val batchSize = items.length / _parallelism
    for (i <- 0 to items.length by batchSize) {
      batches.append(items.slice(i, i + batchSize))
    }
    batches.toArray
  }

  private def getFieldsAndFillable(df: DataFrame, columnList: List[String], statistics: String): DataFrame = {

    //    batch the columns if over 30
    val x = if (columnList.size > 30) {
      val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(_parallelism))
      if (statistics.isEmpty) {
        val colBatches = getBatches(columnList).par
        colBatches.tasksupport = taskSupport
        colBatches.map { batch =>
          df.select(batch map col: _*)
            .summary()
            .select("Summary" +: batch map col: _*)
        }.seq.toArray.reduce((x, y) => x.join(broadcast(y), Seq("Summary")))

      } else {
        val colBatches = getBatches(columnList).par
        colBatches.tasksupport = taskSupport
        colBatches.map { batch =>
          df.select(batch map col: _*)
            .summary(statistics.replaceAll(" ", "").split(","): _*)
            .select("Summary" +: batch map col: _*)
        }.seq.toArray.reduce((x, y) => x.join(broadcast(y), Seq("Summary")))


        //      df.summary(statistics.replaceAll(" ", "").split(","): _*)
        //        .select(selectionColumns map col: _*)
      }
    } else { // Don't batch since < 30 cols
      val selectionColumns = "Summary" +: columnList
      if (statistics.isEmpty) df.summary().select(selectionColumns map col: _*)
      else df.summary(statistics.replaceAll(" ", "").split(","): _*).select(selectionColumns map col: _*)
    }
    x
  }

  private def assemblePayload(df: DataFrame, fieldList: List[String], filterCondition: String): Array[(String, Any)] = {

    val summaryStats = getFieldsAndFillable(df, fieldList, filterCondition)
      .drop(col("Summary"))
    val summaryColumns = summaryStats.columns
    val summaryValues = summaryStats.collect()(0).toSeq.toArray
    summaryColumns.zip(summaryValues)
  }

  private def fillMissing(df: DataFrame): NaFillConfig = {

    val (numericFields, characterFields, dateFields, timeFields) = extractTypes(df, _labelCol, _fieldsToIgnoreInVector)

    val numericMapping = if (numericFields.nonEmpty) {
      val numericPayload = assemblePayload(df, numericFields, metricConversion(_numericFillStat)).par

      val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(_parallelism))
      numericPayload.tasksupport = taskSupport

      val numericFilterBuffer = new ArrayBuffer[(String, Double)]()
      numericPayload.map(x => x._1 match {
        case x._1 if x._1 != _labelCol => try {
          numericFilterBuffer += ((x._1, x._2.toString.toDouble))
        } catch {
          case _: Exception => None
        }
        case _ => None
      })

      numericFilterBuffer.toArray.toMap

    } else new ArrayBuffer[(String, Double)]().toArray.toMap

    val characterMapping = if (characterFields.nonEmpty) {
      val characterPayload = assemblePayload(df, characterFields, metricConversion(_characterFillStat))

      val characterFilterBuffer = new ArrayBuffer[(String, String)]

      characterPayload.map(x => x._1 match {
        case x._1 if x._1 != _labelCol => try {
          characterFilterBuffer += ((x._1, x._2.toString))
        } catch {
          case _: Exception => None
        }
        case _ => None
      })
      characterFilterBuffer.toArray.toMap
    } else new ArrayBuffer[(String, String)]().toArray.toMap

    NaFillConfig(
      numericColumns = numericMapping,
      categoricalColumns = characterMapping
    )

  }

  def decideModel(): String = {
    val uniqueLabelCounts = data.
      select(approx_count_distinct(_labelCol, rsd = _filterPrecision))
      .rdd.map(row => row.getLong(0)).take(1)(0)
    val decision = uniqueLabelCounts match {
      case x if x <= _modelSelectionDistinctThreshold => "classifier"
      case _ => "regressor"
    }
    decision
  }

  def generateCleanData(): (DataFrame, NaFillConfig, String) = {

    val preFilter = refactorLabel(data, _labelCol)

    val fillMap = fillMissing(preFilter)
    val filledData = preFilter.na.fill(fillMap.numericColumns).na.fill(fillMap.categoricalColumns)

    (filledData, fillMap, decideModel())

  }

}