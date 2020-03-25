package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.{DataValidation, SchemaUtils}
import com.databricks.labs.automl.utils.data.CategoricalHandler
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class FeaturePipeline(data: DataFrame, isInferenceRun: Boolean = false)
    extends DataValidation {

  private var _labelCol = "label"
  private var _featureCol = "features"
  private var _dateTimeConversionType = "split"

  private var _cardinalityType: String = "exact"
  private var _cardinalityLimit: Int = 200
  private var _cardinalityPrecision: Double = 0.05
  private var _cardinalityCheckMode: String = "silent"
  private var _cardinalityCheckSwitch: Boolean = true

  private val logger: Logger = Logger.getLogger(this.getClass)

  final private val _dataFieldNames = data.schema.fieldNames

  def setLabelCol(value: String): this.type = {
    if (!isInferenceRun)
      assert(
        _dataFieldNames.contains(value),
        s"Label field $value is not in DataFrame!"
      )
    _labelCol = value
    this
  }

  def setFeatureCol(value: String): this.type = {
    _featureCol = value
    this
  }

  def setDateTimeConversionType(value: String): this.type = {
    assert(
      _allowableDateTimeConversions.contains(value),
      s"Supplied conversion type '$value' is not in: " +
        s"${invalidateSelection(value, _allowableDateTimeConversions)}"
    )
    _dateTimeConversionType = value
    this
  }

  def setCardinalityType(value: String): this.type = {
    assert(
      _allowableCardinalilties.contains(value),
      s"Supplied CardinalityType '$value' is not in: " +
        s"${invalidateSelection(value, _allowableCardinalilties)}"
    )
    _cardinalityType = value
    this
  }
  def setCardinalityLimit(value: Int): this.type = {
    require(value > 0, s"Cardinality limit must be greater than 0")
    _cardinalityLimit = value
    this
  }

  def setCardinalityPrecision(value: Double): this.type = {
    require(value >= 0.0, s"Precision must be greater than or equal to 0.")
    require(value <= 1.0, s"Precision must be less than or equal to 1.")
    _cardinalityPrecision = value
    this
  }

  def setCardinalityCheckMode(value: String): this.type = {
    assert(
      _allowableCategoricalFilterModes.contains(value),
      s"${invalidateSelection(value, _allowableCategoricalFilterModes)}"
    )
    _cardinalityCheckMode = value
    this
  }

  def setCardinalityCheck(value: Boolean): this.type = {
    _cardinalityCheckSwitch = value
    this
  }

  def getLabelCol: String = _labelCol

  def getFeatureCol: String = _featureCol

  def getDateTimeConversionType: String = _dateTimeConversionType

  def getCardinalityType: String = _cardinalityType
  def getCardinalityLimit: Int = _cardinalityLimit
  def getCardinalityPrecision: Double = _cardinalityPrecision
  def getCardinalityCheckMode: String = _cardinalityCheckMode
  def getCardinalitySwitchSetting: Boolean = _cardinalityCheckSwitch

  /**
    * Public method for creating a feature vector.
    * Tasks that are covered:
    *   1. Checking types and ensuring that the label column specified in the config is present in the DataFrame
    *   2. Separating numeric types from categorical types
    *   3. Perform validation on categorical types for cardinality checks.
    *   4. String Index available fields
    *   5. Convert DateTime fields to numeric types
    *   6. Assemble the indexers into a vector assembler to create the feature vector
    * @param ignoreList Fields in the DataFrame to ignore for processing
    * @return The Dataframe with a feature vector.
    */
  def makeFeaturePipeline(
    ignoreList: Array[String]
  ): (DataFrame, Array[String], Array[String]) = {

    val dfSchema = data.schema
    if (!isInferenceRun)
      assert(
        dfSchema.fieldNames.contains(_labelCol),
        s"Dataframe does not contain label column named: ${_labelCol}"
      )

    // Extract all of the field types
    val fields = SchemaUtils.extractTypes(data, _labelCol, ignoreList)

    val fieldsToConvertExclusionsSet =
      fields.categoricalFields.filterNot(ignoreList.contains)

    val validatedStringFields =
      validateCardinality(data, fieldsToConvertExclusionsSet)

    // Support exclusions of fields
    val excludedFieldsReady =
      fields.numericFields.filterNot(ignoreList.contains)

    val excludedFieldsToConvert = fields.categoricalFields
      .filterNot(x => ignoreList.contains(x))
      .filterNot(x => validatedStringFields.invalidFields.contains(x))

    // Restrict the fields based on the configured cardinality limits.
    // Depending on settings:
    //    Silent mode - will silently remove the fields that are above the cardinality limit
    //    Warn mode - an exception will be thrown if the cardinality is too high.

    val cardinalityValidatedConversionFields = if (_cardinalityCheckSwitch) {
      if (excludedFieldsToConvert.nonEmpty) {
        new CategoricalHandler(data, _cardinalityCheckMode)
          .setCardinalityType(_cardinalityType)
          .setPrecision(_cardinalityPrecision)
          .validateCategoricalFields(excludedFieldsToConvert, _cardinalityLimit)
          .toList
      } else excludedFieldsToConvert
    } else excludedFieldsToConvert

    val excludedDateFields = fields.dateFields.filterNot(ignoreList.contains)

    val excludedTimeFields = fields.timeFields.filterNot(ignoreList.contains)

    // Modify the Dataframe for datetime / date types
    val (dateTimeModData, dateTimeFields) = convertDateAndTime(
      data,
      excludedDateFields,
      excludedTimeFields,
      _dateTimeConversionType
    )

    // Concatenate the numeric field listing with the new numeric converted datetime fields
    val mergedFields = excludedFieldsReady ++ dateTimeFields

    val (indexers, assembledColumns, assembler) =
      generateAssembly(
        mergedFields,
        cardinalityValidatedConversionFields,
        _featureCol
      )

    val createPipe = new Pipeline()
      .setStages(indexers :+ assembler)

    val fieldsToInclude = if (!isInferenceRun) {
      assembledColumns ++ Array(_featureCol, _labelCol)
    } else {
      assembledColumns ++ Array(_featureCol) ++ ignoreList
    }

    //DEBUG
    logger.log(
      Level.DEBUG,
      s" MAKE FEATURE PIPELINE FIELDS TO INCLUDE: ${fieldsToInclude.mkString(", ")}"
    )

    val transformedData = createPipe
      .fit(dateTimeModData)
      .transform(dateTimeModData)
      .select(fieldsToInclude ++ ignoreList map col: _*)

    val transformedExtract = if (fields.categoricalFields.contains(_labelCol)) {
      transformedData
        .drop(_labelCol)
        .withColumnRenamed(s"${_labelCol}_si", _labelCol)
    } else {
      transformedData
    }

    val assembledColumnsOutput =
      if (fields.categoricalFields.contains(_labelCol)) {
        assembledColumns.filterNot(x => x.contains(s"${_labelCol}_si"))
      } else assembledColumns

    val fieldsToIncludeOutput =
      if (fields.categoricalFields.contains(_labelCol)) {
        fieldsToInclude.filterNot(x => x.contains(s"${_labelCol}_si"))
      } else fieldsToInclude

    (
      transformedExtract,
      assembledColumnsOutput,
      fieldsToIncludeOutput.filterNot(_.contains(_featureCol))
    )

  }

  def applyOneHotEncoding(
    featureColumns: Array[String],
    totalFields: Array[String]
  ): (DataFrame, Array[String], Array[String]) = {

    // From the featureColumns collection, get the string indexed fields.
    val stringIndexedFields = featureColumns
      .filter(x => x.takeRight(3) == "_si")
      .filterNot(x => x.contains(_labelCol))

    // Get the fields that are not String Indexed.
    val remainingFeatureFields =
      featureColumns.filterNot(x => x.takeRight(3) == "_si")

    // Drop the feature field that has already been created.
    val adjustedData =
      if (data.schema.fieldNames.contains(_featureCol)) data.drop(_featureCol)
      else data

    // One hot encode the StringIndexed fields, if present and generate the feature vector.
    val (outputData, featureFields) = if (stringIndexedFields.length > 0) {

      val (encoder, encodedColumns) = oneHotEncodeStrings(
        stringIndexedFields.toList
      )

      val fullFeatureColumns = remainingFeatureFields ++ encodedColumns

      val assembler = new VectorAssembler()
        .setInputCols(fullFeatureColumns)
        .setOutputCol(_featureCol)

      val pipe = new Pipeline()
        .setStages(Array(encoder) :+ assembler)

      val transformedData = pipe.fit(adjustedData).transform(adjustedData)

      (transformedData, fullFeatureColumns)

    } else {

      val assembler = new VectorAssembler()
        .setInputCols(featureColumns)
        .setOutputCol(_featureCol)

      val pipe = new Pipeline()
        .setStages(Array(assembler))

      val transformedData = pipe.fit(adjustedData).transform(adjustedData)

      (transformedData, featureColumns)

    }

    val fullFinalSchema = outputData.schema.fieldNames.diff(stringIndexedFields)

    val dataReturn = outputData.select(fullFinalSchema map col: _*)

    val dataSchema = fullFinalSchema.filterNot(_.contains(_featureCol))

    //DEBUG
    logger.log(
      Level.DEBUG,
      s" Post OneHotEncoding Fields: ${fullFinalSchema.mkString(", ")}"
    )

    (dataReturn, featureFields, dataSchema)

  }

}
