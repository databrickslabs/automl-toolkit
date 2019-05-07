package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.inference.InferenceConfig._
import com.databricks.labs.automl.utils.DataValidation
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


class FeaturePipeline(data: DataFrame, isInferenceRun: Boolean = false) extends DataValidation {

  private var _labelCol = "label"
  private var _featureCol = "features"
  private var _dateTimeConversionType = "split"
  private val logger: Logger = Logger.getLogger(this.getClass)

  final private val _dataFieldNames = data.schema.fieldNames


  def setLabelCol(value: String): this.type = {
    if (!isInferenceRun) assert(_dataFieldNames.contains(value), s"Label field $value is not in DataFrame!")
    _labelCol = value
    this
  }

  def setFeatureCol(value: String): this.type = {
    _featureCol = value
    this
  }

  def setDateTimeConversionType(value: String): this.type = {
    assert(_allowableDateTimeConversions.contains(value), s"Supplied conversion type '$value' is not in: " +
      s"${invalidateSelection(value, _allowableDateTimeConversions)}")
    _dateTimeConversionType = value
    this
  }

  def getLabelCol: String = _labelCol

  def getFeatureCol: String = _featureCol

  def getDateTimeConversionType: String = _dateTimeConversionType

  def makeFeaturePipeline(ignoreList: Array[String]): (DataFrame, Array[String], Array[String]) = {

    val dfSchema = data.schema
    if(!isInferenceRun) assert(dfSchema.fieldNames.contains(_labelCol), s"Dataframe does not contain label column named: ${_labelCol}")

    // Extract all of the field types
    val (fieldsReady, fieldsToConvert, dateFields, timeFields) = extractTypes(data, _labelCol, ignoreList)

    // Support exclusions of fields
    val excludedFieldsReady = fieldsReady.filterNot(x => ignoreList.contains(x))

    val excludedFieldsToConvert = fieldsToConvert.filterNot(x => ignoreList.contains(x))

    val excludedDateFields = dateFields.filterNot(x => ignoreList.contains(x))

    val excludedTimeFields = timeFields.filterNot(x => ignoreList.contains(x))

    // Modify the Dataframe for datetime / date types
    val (dateTimeModData, dateTimeFields) = convertDateAndTime(data, excludedDateFields,
      excludedTimeFields, _dateTimeConversionType)

    // Concatenate the numeric field listing with the new numeric converted datetime fields
    val mergedFields = excludedFieldsReady ++ dateTimeFields

    val (indexers, assembledColumns, assembler) = generateAssembly(mergedFields, excludedFieldsToConvert, _featureCol)

    val createPipe = new Pipeline()
      .setStages(indexers :+ assembler)

    val fieldsToInclude = if (!isInferenceRun) {
      assembledColumns ++ Array(_featureCol, _labelCol) ++ ignoreList
    } else {
      assembledColumns ++ Array(_featureCol) ++ ignoreList
    }


    //DEBUG
    logger.log(Level.DEBUG, s" MAKE FEATURE PIPELINE FIELDS TO INCLUDE: ${fieldsToInclude.mkString(", ")}")

    val transformedData = createPipe.fit(dateTimeModData).transform(dateTimeModData).select(fieldsToInclude map col:_*)

    val transformedExtract = if(fieldsToConvert.contains(_labelCol)){
      transformedData
        .drop(_labelCol)
        .withColumnRenamed(s"${_labelCol}_si", _labelCol)
    } else {
      transformedData
    }

    val assembledColumnsOutput = if(fieldsToConvert.contains(_labelCol)) {
      assembledColumns.filterNot(x => x.contains(s"${_labelCol}_si"))
    } else assembledColumns

    val fieldsToIncludeOutput = if(fieldsToConvert.contains(_labelCol)) {
      fieldsToInclude.filterNot(x => x.contains(s"${_labelCol}_si"))
    } else fieldsToInclude

    (transformedExtract, assembledColumnsOutput, fieldsToIncludeOutput.filterNot(_.contains(_featureCol)))

  }

  def applyOneHotEncoding(featureColumns: Array[String], totalFields: Array[String]):
  (DataFrame, Array[String], Array[String]) = {

    // From the featureColumns collection, get the string indexed fields.
    val stringIndexedFields = featureColumns.filter(x => x.takeRight(3) == "_si").filterNot(x => x.contains(_labelCol))

    // Get the fields that are not String Indexed.
    val remainingFeatureFields = featureColumns.filterNot(x => x.takeRight(3) == "_si")

    // Drop the feature field that has already been created.
    val adjustedData = if (data.schema.fieldNames.contains(_featureCol)) data.drop(_featureCol) else data

    // One hot encode the StringIndexed fields, if present and generate the feature vector.
    val (outputData, featureFields) = if(stringIndexedFields.length > 0) {

      val (encoder, encodedColumns) = oneHotEncodeStrings(stringIndexedFields.toList)

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

    val dataReturn = outputData.select(fullFinalSchema map col:_*)

    val dataSchema = fullFinalSchema.filterNot(_.contains(_featureCol))

    //DEBUG
    logger.log(Level.DEBUG, s" Post OneHotEncoding Fields: ${fullFinalSchema.mkString(", ")}")

    (dataReturn, featureFields, dataSchema)

  }

}
