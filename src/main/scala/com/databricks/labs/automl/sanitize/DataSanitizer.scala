package com.databricks.labs.automl.sanitize

import com.databricks.labs.automl.exceptions.BooleanFieldFillException
import com.databricks.labs.automl.inference.{NaFillConfig, NaFillPayload}
import com.databricks.labs.automl.utils.structures.FeatureEngineeringEnums.FeatureEngineeringEnums
import com.databricks.labs.automl.utils.structures.{
  FeatureEngineeringAllowables,
  FeatureEngineeringEnums
}
import com.databricks.labs.automl.utils.{DataValidation, SchemaUtils}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable.ArrayBuffer

class DataSanitizer(data: DataFrame) extends DataValidation {

  private var _labelCol = "label"
  private var _featureCol = "features"
  private var _numericFillStat = "mean"
  private var _characterFillStat = "max"
  private var _modelSelectionDistinctThreshold = 10
  private var _fieldsToIgnoreInVector = Array.empty[String]
  private var _filterPrecision: Double = 0.01
  private var _parallelism: Int = 20

  private var _categoricalNAFillMap: Map[String, String] =
    Map.empty[String, String]
  private var _numericNAFillMap: Map[String, AnyVal] = Map.empty[String, AnyVal]
  private var _characterNABlanketFill: String = ""
  private var _numericNABlanketFill: Double = 0.0
  private var _naFillMode: String = "auto"

  final private val _allowableNAFillModes: List[String] =
    List(
      "auto",
      "mapFill",
      "blanketFillAll",
      "blanketFillCharOnly",
      "blanketFillNumOnly"
    )

  def setLabelCol(value: String): this.type = {
    _labelCol = value
    this
  }

  def setFeatureCol(value: String): this.type = {
    _featureCol = value
    this
  }

  def setNumericFillStat(value: String): this.type = {
    _numericFillStat = value
    this
  }

  def setCharacterFillStat(value: String): this.type = {
    _characterFillStat = value
    this
  }

  def setModelSelectionDistinctThreshold(value: Int): this.type = {
    _modelSelectionDistinctThreshold = value
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
    if (value == 0.0)
      println(
        "Warning! Precision of 0 is an exact calculation of quantiles and may not be performant!"
      )
    _filterPrecision = value
    this
  }

  def setCategoricalNAFillMap(value: Map[String, String]): this.type = {
    _categoricalNAFillMap = value
    this
  }

  def setNumericNAFillMap(value: Map[String, AnyVal]): this.type = {
    _numericNAFillMap = value
    this
  }

  def setCharacterNABlanketFillValue(value: String): this.type = {
    _characterNABlanketFill = value
    this
  }

  def setNumericNABlanketFillValue(value: Double): this.type = {
    _numericNABlanketFill = value
    this
  }

  /**
    * Setter for determining the fill mode for handling na values.
    *
    * @param value Mode for na fill<br>
    *                Available modes: <br>
    *                  <i>auto</i> : Stats-based na fill for fields.  Usage of .setNumericFillStat and
    *                  .setCharacterFillStat will inform the type of statistics that will be used to fill.<br>
    *                  <i>mapFill</i> : Custom by-column overrides to 'blanket fill' na values on a per-column
    *                  basis.  The categorical (string) fields are set via .setCategoricalNAFillMap while the
    *                  numeric fields are set via .setNumericNAFillMap.<br>
    *                  <i>blanketFillAll</i> : Fills all fields based on the values specified by
    *                  .setCharacterNABlanketFillValue and .setNumericNABlanketFillValue.  All NA's for the
    *                  appropriate types will be filled in accordingly throughout all columns.<br>
    *                  <i>blanketFillCharOnly</i> Will use statistics to fill in numeric fields, but will replace
    *                  all categorical character fields na values with a blanket fill value. <br>
    *                  <i>blanketFillNumOnly</i> Will use statistics to fill in character fields, but will replace
    *                  all numeric fields na values with a blanket value.
    * @author Ben Wilson, Databricks
    * @since 0.5.2
    * @throws IllegalArgumentException if mode is not supported
    */
  @throws(classOf[IllegalArgumentException])
  def setNAFillMode(value: String): this.type = {
    require(
      _allowableNAFillModes.contains(value),
      s"NA fill mode $value is not supported. Must be one of : " +
        s"${_allowableNAFillModes.mkString(", ")}"
    )
    _naFillMode = value
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

  def getCategoricalNAFillMap: Map[String, String] = _categoricalNAFillMap

  def getNumericNAFillMap: Map[String, AnyVal] = _numericNAFillMap

  def getCharacterNABlanketFillValue: String = _characterNABlanketFill

  def getNumericNABlanketFillValue: Double = _numericNABlanketFill

  def getNaFillMode: String = _naFillMode

  private var _labelValidation: Boolean = false

  def labelValidationOn(): this.type = {
    _labelValidation = true
    this
  }

  private def convertLabel(df: DataFrame): DataFrame = {

    val stringIndexer = getLabelIndexer(df)

    stringIndexer
      .fit(data)
      .transform(data)
      .withColumn(this._labelCol, col(s"${this._labelCol}_si"))
      .drop(this._labelCol + "_si")
  }

  def getLabelIndexer(df: DataFrame): StringIndexer = {
    new StringIndexer()
      .setInputCol(this._labelCol)
      .setOutputCol(this._labelCol + "_si")
  }

  private def refactorLabel(df: DataFrame, labelColumn: String): DataFrame = {

    SchemaUtils
      .extractSchema(df.schema)
      .foreach(
        x =>
          x.fieldName match {
            case `labelColumn` =>
              x.dataType match {
                case StringType  => labelValidationOn()
                case BooleanType => labelValidationOn()
                case BinaryType  => labelValidationOn()
                case _           => None
              }
            case _ => None
        }
      )
    if (_labelValidation) convertLabel(df) else df
  }

  private def metricConversion(metric: String): String = {

    val allowableFillArray = Array("min", "25p", "mean", "median", "75p", "max")

    assert(
      allowableFillArray.contains(metric),
      s"The metric supplied, '$metric' is not in: " +
        s"${invalidateSelection(metric, allowableFillArray)}"
    )

    val summaryMetric = metric match {
      case "25p"    => "25%"
      case "median" => "50%"
      case "75p"    => "75%"
      case _        => metric
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

  private def getFieldsAndFillable(df: DataFrame,
                                   columnList: List[String],
                                   statistics: String): DataFrame = {

    val dfParts = df.rdd.partitions.length.toDouble
//    val summaryParts = Math.min(Math.ceil(dfParts / 20.0).toInt, 200)
    val summaryParts =
      Math.max(32, Math.min(Math.ceil(dfParts / 20.0).toInt, 200))
    val selectionColumns = "Summary" +: columnList
    val x = if (statistics.isEmpty) {
      val colBatches = getBatches(columnList)
      colBatches
        .map { batch =>
          df.coalesce(summaryParts)
            .select(batch map col: _*)
            .summary()
            .select("Summary" +: batch map col: _*)
        }
        .seq
        .toArray
        .reduce((x, y) => x.join(broadcast(y), Seq("Summary")))

    } else {
      df.coalesce(summaryParts)
        .summary(statistics.replaceAll(" ", "").split(","): _*)
        .select(selectionColumns map col: _*)
    }
    x
  }

  private def assemblePayload(df: DataFrame,
                              fieldList: List[String],
                              filterCondition: String): Array[(String, Any)] = {

    val summaryStats = getFieldsAndFillable(df, fieldList, filterCondition)
      .drop(col("Summary"))
    val summaryColumns = summaryStats.columns
    val summaryValues = summaryStats.collect()(0).toSeq.toArray
    summaryColumns.zip(summaryValues)
  }

  private def getCategoricalFillType(value: String): FeatureEngineeringEnums = {

    value match {
      case "min" => FeatureEngineeringEnums.MIN
      case "max" => FeatureEngineeringEnums.MAX
    }

  }

  /**
    * Boolean filling based on the filterCondition for categorical data (min or max)
    * @param df DataFrame containing BooleanType fields that may have null values
    * @param fieldList List of the Boolean Fields
    * @param filterCondition The setting of whether to use min or max values based on the data to fill na's
    * @return The fill config data for the Boolean type columns consisting of ColumnName, Boolean fill value
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    * @throws BooleanFieldFillException if the mode setting for selecting the fill value is not supported.
    */
  @throws(classOf[BooleanFieldFillException])
  private def getBooleanFill(
    df: DataFrame,
    fieldList: List[String],
    filterCondition: String
  ): Array[(String, Boolean)] = {

    val filterSelection = getCategoricalFillType(filterCondition)

    fieldList
      .map(x => {
        val booleanFieldStats =
          df.select(x)
            .groupBy(x)
            .agg(count(col(x)).alias(FeatureEngineeringEnums.COUNT_COL.value))

        val sortedStats = filterSelection match {
          case FeatureEngineeringEnums.MIN =>
            booleanFieldStats
              .orderBy(col(FeatureEngineeringEnums.COUNT_COL.value).asc)
              .head(1)
          case FeatureEngineeringEnums.MAX =>
            booleanFieldStats
              .orderBy(col(FeatureEngineeringEnums.COUNT_COL.value).desc)
              .head(1)
          case _ =>
            throw BooleanFieldFillException(
              x,
              filterCondition,
              FeatureEngineeringAllowables.ALLOWED_CATEGORICAL_FILL_MODES.values
            )
        }
        (x, sortedStats.head.getBoolean(0))

      })
      .toArray

  }

  /**
    * Helper method for extraction the fields types based on the schema and calculating the statistics to be used to
    * determine fill values for the columns.
    *
    * @param df A DataFrame that has already had the label field converted to the appropriate (Double) Type
    * @return NaFillConfig : A mapping for numeric and string fields that represents the values to put in for each column.
    * @since 0.1.0
    * @author Ben Wilson, Databricks
    */
  private def payloadExtraction(df: DataFrame): NaFillPayload = {

    val typeExtract =
      SchemaUtils.extractTypes(df, _labelCol, _fieldsToIgnoreInVector)

    val numericPayload =
      assemblePayload(
        df,
        typeExtract.numericFields,
        metricConversion(_numericFillStat)
      )
    val characterPayload =
      assemblePayload(
        df,
        typeExtract.categoricalFields,
        metricConversion(_characterFillStat)
      )
    val booleanPayload = getBooleanFill(
      df,
      typeExtract.booleanFields,
      metricConversion(_characterFillStat)
    )

    NaFillPayload(characterPayload, numericPayload, booleanPayload)

  }

  /**
    * Helper method for ensuring that the label column isn't overridden
    *
    * @param payload Array of field name, value for overriding of numeric fields.
    * @return Map of Field name to fill value converted to Double type.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  private def numericMapper(
    payload: Array[(String, Any)]
  ): Map[String, Double] = {

    val buffer = new ArrayBuffer[(String, Double)]

    payload.map(
      x =>
        x._1 match {
          case x._1 if x._1 != _labelCol =>
            try {
              buffer += ((x._1, x._2.toString.toDouble))
            } catch {
              case _: Exception => None
            }
          case _ => None
      }
    )
    buffer.toArray.toMap
  }

  /**
    * Helper method for ensuring that the label column isn't overridden
    *
    * @param payload Array of field name, value for overriding character fields.
    * @return Map of Field name to fill value convert to String type.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  private def characterMapper(
    payload: Array[(String, Any)]
  ): Map[String, String] = {

    val buffer = new ArrayBuffer[(String, String)]

    payload.map(
      x =>
        x._1 match {
          case x._1 if x._1 != _labelCol =>
            try {
              buffer += ((x._1, x._2.toString))
            } catch {
              case _: Exception => None
            }
          case _ => None
      }
    )

    buffer.toArray.toMap

  }

  /**
    * Helper method for generating a statistics-based approach for calculating 'smart fillable' values for na's in the
    * feature vector fields.
    *
    * @param df A DataFrame that has already had the label field converted to the appropriate (Double) Type
    * @return NaFillConfig : A mapping for numeric and string fields that represents the values to put in for each column.
    * @since 0.1.0
    * @author Ben Wilson, Databricks
    */
  private def fillMissing(df: DataFrame): NaFillConfig = {

    val payloads = payloadExtraction(df)

    val numericMapping = numericMapper(payloads.numeric)

    val characterMapping = characterMapper(payloads.categorical)

    NaFillConfig(
      numericColumns = numericMapping,
      categoricalColumns = characterMapping,
      booleanColumns = payloads.boolean.toMap
    )

  }

  /**
    * Private method for applying a full blanket na fill on all fields to be involved in the feature vector.
    *
    * @param df A DataFrame that has already had the label field converted to the appropriate (Double) Type
    * @return NaFillConfig : A mapping for numeric and string fields that represents the values to put in for each column.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  private def blanketNAFill(df: DataFrame): NaFillConfig = {

    val payloadTypes =
      SchemaUtils.extractTypes(df, _labelCol, _fieldsToIgnoreInVector)

    val characterBuffer = new ArrayBuffer[(String, Any)]
    val numericBuffer = new ArrayBuffer[(String, Any)]

    payloadTypes.numericFields.foreach(
      x => numericBuffer += ((x, _numericNABlanketFill))
    )
    payloadTypes.categoricalFields.foreach(
      x => characterBuffer += ((x, _characterNABlanketFill))
    )

    //TODO: update Boolean overrides.
    NaFillConfig(
      characterMapper(characterBuffer.toArray),
      numericMapper(numericBuffer.toArray),
      payloadTypes.booleanFields.map(x => (x, false)).toMap
    )

  }

  /**
    * Private method for applying a full blanket na fill on only character fields to be involved in the feature vector.
    * Numeric fields will use the stats mode defined in .setNumericFillStat
    *
    * @param df A DataFrame that has already had the label field converted to the appropriate (Double) Type
    * @return NaFillConfig : A mapping for numeric and string fields that represents the values to put in for each column.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  private def blanketFillCharOnly(df: DataFrame): NaFillConfig = {

    val payloads = fillMissing(df)

    val buffer = new ArrayBuffer[(String, String)]

    payloads.categoricalColumns.map(
      x => buffer += ((x._1, _characterNABlanketFill))
    )

    NaFillConfig(
      characterMapper(buffer.toArray),
      payloads.numericColumns,
      payloads.booleanColumns
    )

  }

  /**
    * Private method for applying a full blanket na fill on only numeric fields to be involved in the feature vector.
    * Character fields will use the stats mode defined in .setCharacterFillStat
    *
    * @param df A DataFrame that has already had the label field converted to the appropriate (Double) Type
    * @return NaFillConfig : A mapping for numeric and string fields that represents the values to put in for each column.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  private def blanketFillNumOnly(df: DataFrame): NaFillConfig = {

    val payloads = fillMissing(df)

    val buffer = new ArrayBuffer[(String, Double)]

    payloads.numericColumns.map(x => buffer += ((x._1, _numericNABlanketFill)))

    NaFillConfig(
      payloads.categoricalColumns,
      numericMapper(buffer.toArray),
      payloads.booleanColumns
    )

  }

  /**
    * Validation run-time check for supplied maps, if used.
    *
    * @param df A DataFrame that has already had the label field converted to the appropriate (Double) Type
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    * @throws IllegalArgumentException if a map value refers to a column not in the dataset
    * @throws UnsupportedOperationException if no map overrides have been specified in the run configuration
    */
  @throws(classOf[UnsupportedOperationException])
  @throws(classOf[IllegalArgumentException])
  private def validateMapSchemaMembership(df: DataFrame): Unit = {
    val suppliedSchema = df.schema.names

    if (_numericNAFillMap.nonEmpty)
      _numericNAFillMap.keys.foreach(
        x =>
          require(
            suppliedSchema.contains(x),
            s"Field $x supplied in .setNumericNAFillMap() is not a valid column name in the DataFrame."
        )
      )

    if (_categoricalNAFillMap.nonEmpty)
      _categoricalNAFillMap.keys.foreach(
        x =>
          require(
            suppliedSchema.contains(x),
            s"Field $x supplied in .setCategoricalNAFillMap() is not a valid column name in the DataFrame."
        )
      )
    if (_categoricalNAFillMap.isEmpty && _numericNAFillMap.isEmpty)
      throw new UnsupportedOperationException(
        s"Map Fill mode has been defined for NA Fill but " +
          s"no map overrides have been specified.  Check configuration and ensure that either categoricalNAFillMap " +
          s"or numericNAFillMap values have been set."
      )
  }

  /**
    * Private method for submitting a Map of categorical and numeric overrides for na fill based on column name -> value
    * as set in .setCategoricalNAFillMap and .setNumericNAFillMap any fields not included in these maps will use the
    * statistics-based approaches to fill na's.
    *
    * @param df A DataFrame that has already had the label field converted to the appropriate (Double) Type
    * @return NaFillConfig : A mapping for numeric and string fields that represents the values to put in for each column.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  private def mapNAFill(df: DataFrame): NaFillConfig = {

    validateMapSchemaMembership(df)

    val payloads = fillMissing(df)

    val numBuffer = new ArrayBuffer[(String, Double)]
    val charBuffer = new ArrayBuffer[(String, String)]

    payloads.categoricalColumns.map(
      x =>
        x._1 match {
          case x._1 if _categoricalNAFillMap.contains(x._1) =>
            charBuffer += ((x._1, _categoricalNAFillMap(x._1).toString))
          case _ => charBuffer += x
      }
    )
    payloads.numericColumns.map(
      x =>
        x._1 match {
          case x._1 if _numericNAFillMap.contains(x._1) =>
            numBuffer += ((x._1, _numericNAFillMap(x._1).toString.toDouble))
          case _ => numBuffer += x
      }
    )

    NaFillConfig(
      characterMapper(charBuffer.toArray),
      numericMapper(numBuffer.toArray),
      payloads.booleanColumns.map(x => x._1 -> false)
    )

  }

  /**
    * Private method for handling control logic depending on na fill mode selected
    *
    * @param df A DataFrame that has already had the label field converted to the appropriate (Double) Type
    * @return NaFillConfig : A mapping for numeric and string fields that represents the values to put in for each column.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  private def fillNA(df: DataFrame): NaFillConfig = {

    _naFillMode match {
      case "auto" => fillMissing(df)
      case "blanketFillAll" =>
        blanketNAFill(df)
      case "blanketFillCharOnly" => blanketFillCharOnly(df)
      case "blanketFillNumOnly"  => blanketFillNumOnly(df)
      case "mapFill"             => mapNAFill(df)
      case _ =>
        throw new UnsupportedOperationException(
          s"The naFill Mode ${_naFillMode} is not supported. " +
            s"Must be one of: ${_allowableNAFillModes.mkString(", ")}"
        )
    }

  }

  def decideModel(): String = {
    val uniqueLabelCounts = data
      .select(approx_count_distinct(_labelCol, rsd = _filterPrecision))
      .rdd
      .map(row => row.getLong(0))
      .take(1)(0)
    val decision = uniqueLabelCounts match {
      case x if x <= _modelSelectionDistinctThreshold => "classifier"
      case _                                          => "regressor"
    }
    decision
  }

  def generateCleanData(
    naFillConfig: NaFillConfig = null,
    refactorLabelFlag: Boolean = true,
    decidedModel: String = ""
  ): (DataFrame, NaFillConfig, String) = {

    val preFilter = if (refactorLabelFlag) {
      refactorLabel(data, _labelCol)
    } else {
      data
    }

    val fillMap = if (naFillConfig == null) {
      fillNA(preFilter)
    } else {
      naFillConfig
    }
    val filledData = preFilter.na
      .fill(fillMap.numericColumns)
      .na
      .fill(fillMap.categoricalColumns)
      .na
      .fill(fillMap.booleanColumns)

    if (decidedModel != null && decidedModel.nonEmpty) {
      (filledData, fillMap, decidedModel)
    } else {
      (filledData, fillMap, decideModel())
    }

  }

}
