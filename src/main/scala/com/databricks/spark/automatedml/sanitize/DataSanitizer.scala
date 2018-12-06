package com.databricks.spark.automatedml.sanitize

import com.databricks.spark.automatedml.utils.DataValidation
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable.ArrayBuffer

object DataSanitizerPythonHelper {
  var _decision: String = _
  def generateCleanData(sanitizer: DataSanitizer, dfName: String): Unit = {
    val (cleanDf, decision) = sanitizer.generateCleanData()
    _decision = decision
    cleanDf.createOrReplaceTempView(dfName)
    println("Dataframe has been cleaned and registered as " + dfName + " " + cleanDf)
    println("Model decision was for " + decision)
  }

  def getModelDecision: String = {
    _decision
  }
}

class DataSanitizer(data: DataFrame) extends DataValidation {

  private var _labelCol = "label"
  private var _featureCol = "features"
  private var _numericFillStat = "mean"
  private var _characterFillStat = "max"
  private var _modelSelectionDistinctThreshold = 10

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

  def getLabel: String = _labelCol
  def getFeatureCol: String = _featureCol
  def getNumericFillStat: String = _numericFillStat
  def getCharacterFillStat: String = _characterFillStat
  def getModelSelectionDistinctThreshold: Int = _modelSelectionDistinctThreshold

  private def convertLabel(df: DataFrame): DataFrame = {

    val stringIndexer = new StringIndexer()
      .setInputCol(this._labelCol)
      .setOutputCol(this._labelCol + "_si")

    stringIndexer.fit(data).transform(data)
      .withColumn(this._labelCol, col(s"${this._labelCol}_si"))
      .drop(this._labelCol + "_si")
  }

  private def refactorLabel(df: DataFrame, labelColumn: String): DataFrame = {

    var validation = false

    extractSchema(df.schema).foreach(x =>
      x._2 match {
        case `labelColumn` => x._1 match {
          case StringType => validation = true
          case BooleanType => validation = true
          case BinaryType => validation = true
          case _ => None
        }
        case _ => None
      })
    if (validation) convertLabel(df) else df
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

  private def getFieldsAndFillable(df: DataFrame, columnList: List[String]): DataFrame = {
    val selectionColumns = "Summary" +: columnList
    df.summary().select(selectionColumns map col:_*)
  }

  private def assemblePayload(df: DataFrame, fieldList: List[String], filterCondition: String): Array[(String, Any)] = {

    val summaryStats = getFieldsAndFillable(df, fieldList)
      .filter(col("Summary") === filterCondition)
      .drop(col("Summary"))
    val summaryColumns = summaryStats.columns
    val summaryValues = summaryStats.collect()(0).toSeq.toArray
    summaryColumns.zip(summaryValues)
  }

  private def fillMissing(df: DataFrame): (Map[String, Double], Map[String, String]) = {

    val (numericFields, characterFields) = extractTypes(df, _labelCol)

    val numericPayload = assemblePayload(df, numericFields, metricConversion(_numericFillStat))
    val characterPayload = assemblePayload(df, characterFields, metricConversion(_characterFillStat))

    val numericFilterBuffer = new ArrayBuffer[(String, Double)]
    val characterFilterBuffer = new ArrayBuffer[(String, String)]

    numericPayload.map(x => x._1 match {
      case x._1 if x._1 != _labelCol => try{numericFilterBuffer += ((x._1, x._2.toString.toDouble))
      } catch {
        case _:Exception => None
      }
      case _ => None
    })

    characterPayload.map(x => x._1 match {
      case x._1 if x._1 != _labelCol => try{characterFilterBuffer += ((x._1, x._2.toString))
      } catch {
        case _:Exception => None
      }
      case _ => None
    })

    val numericMapping = numericFilterBuffer.toArray.toMap

    val characterMapping = characterFilterBuffer.toArray.toMap

    (numericMapping, characterMapping)
  }

  def decideModel(): String = {
    val uniqueLabelCounts = data.select(_labelCol).distinct.count
    val decision = uniqueLabelCounts match {
      case x if x <= _modelSelectionDistinctThreshold => "classifier"
      case _ => "regressor"
    }
    decision
  }

  def generateCleanData(): (DataFrame, String) = {

    val preFilter = refactorLabel(data, _labelCol)

    val (numMap, charMap) = fillMissing(preFilter)
    val filledData = preFilter.na.fill(numMap).na.fill(charMap)

    (filledData, decideModel())

  }

}