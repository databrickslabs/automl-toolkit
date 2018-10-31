package com.databricks.spark.automatedml


import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame


class FeaturePipeline(data: DataFrame) extends DataValidation {

  private var _labelCol = "label"
  private var _featureCol = "features"

  final private val _dataFieldNames = data.schema.fieldNames

  def setLabelCol(value: String): this.type = {
    assert(_dataFieldNames.contains(value), s"Label field $value is not in DataFrame!")
    this._labelCol = value
    this
  }
  def setFeatureCol(value: String): this.type = {
    this._featureCol = value
    this
  }

  def getLabelCol: String = this._labelCol
  def getFeatureCol: String = this._featureCol


  def makeFeaturePipeline(): (DataFrame, Array[String]) = {

    val dfSchema = data.schema
    assert(dfSchema.fieldNames.contains(_labelCol), s"Dataframe does not contain label column named: ${_labelCol}")

    val (fieldsReady, fieldsToConvert) = extractTypes(data, _labelCol)
    val (indexers, assembledColumns, assembler) = generateAssembly(fieldsReady, fieldsToConvert, _featureCol)

    val createPipe = new Pipeline()
      .setStages(indexers :+ assembler)

    (createPipe.fit(data).transform(data), assembledColumns)

  }



}
