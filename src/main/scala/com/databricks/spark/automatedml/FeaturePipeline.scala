package com.databricks.spark.automatedml


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ListBuffer


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

  private def indexStrings(categoricalFields: List[String]): (Array[StringIndexer], Array[String]) = {

    var indexedColumns = new ListBuffer[String]
    var stringIndexers = new ListBuffer[StringIndexer]

    categoricalFields.map(x => {
      val stringIndexedColumnName = x + "_si"
      val stringIndexerObj = new StringIndexer()
        .setHandleInvalid("keep")
        .setInputCol(x)
        .setOutputCol(stringIndexedColumnName)
      indexedColumns += stringIndexedColumnName
      stringIndexers += stringIndexerObj
    })

    (stringIndexers.result.toArray, indexedColumns.result.toArray)

  }

  private def generateAssembly(numericColumns: List[String], characterColumns: List[String]):
  (Array[StringIndexer], Array[String], VectorAssembler) = {

    val assemblerColumns = new ListBuffer[String]
    numericColumns.map(x => assemblerColumns += x)

    val (indexers, indexedColumns) = indexStrings(characterColumns)
    indexedColumns.map(x => assemblerColumns += x)

    val assembledColumns = assemblerColumns.result.toArray

    val assembler = new VectorAssembler()
      .setInputCols(assembledColumns)
      .setOutputCol(_featureCol)

    (indexers, assembledColumns, assembler)
  }

  def makeFeaturePipeline(): (DataFrame, Array[String]) = {

    val dfSchema = data.schema
    assert(dfSchema.fieldNames.contains(_labelCol), s"Dataframe does not contain label column named: ${_labelCol}")

    val (fieldsReady, fieldsToConvert) = extractTypes(data, _labelCol)
    val (indexers, assembledColumns, assembler) = generateAssembly(fieldsReady, fieldsToConvert)

    val createPipe = new Pipeline()
      .setStages(indexers :+ assembler)

    (createPipe.fit(data).transform(data), assembledColumns)

  }



}
