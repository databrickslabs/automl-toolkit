package com.databricks.spark.automatedml.pipeline

import com.databricks.spark.automatedml.utils.DataValidation
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


class FeaturePipeline(data: DataFrame) extends DataValidation {

  private var _labelCol = "label"
  private var _featureCol = "features"
  private var _dateTimeConversionType = "split"

  final private val _dataFieldNames = data.schema.fieldNames


  def setLabelCol(value: String): this.type = {
    assert(_dataFieldNames.contains(value), s"Label field $value is not in DataFrame!")
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

  //TODO: allow for restricted fields to not be included in the featurization vector.
  def makeFeaturePipeline(ignoreList: Array[String] = Array("")): (DataFrame, Array[String]) = {

    val dfSchema = data.schema
    assert(dfSchema.fieldNames.contains(_labelCol), s"Dataframe does not contain label column named: ${_labelCol}")

    // Extract all of the field types
    val (fieldsReady, fieldsToConvert, dateFields, timeFields) = extractTypes(data, _labelCol)

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

    val fieldsToInclude = assembledColumns ++ List(_featureCol, _labelCol)

    val transformedData = createPipe.fit(dateTimeModData).transform(dateTimeModData).select(fieldsToInclude map col:_*)

    (transformedData, assembledColumns)

  }


}
