package com.databricks.spark.automatedml

trait Evolution {

  var _labelCol = "label"
  var _featureCol = "features"
  var _trainPortion = 0.8
  var _kFold = 3
  var _seed = 42L
  var _kFoldIteratorRange: scala.collection.parallel.immutable.ParRange = Range(0, _kFold).par


  def setLabelCol(value: String): this.type = {
    this._labelCol = value
    this
  }

  def setFeaturesCol(value: String): this.type = {
    this._featureCol = value
    this
  }

  def setTrainPortion(value: Double): this.type = {
    assert(value < 1.0 & value > 0.0, "Training portion must be in the range > 0 and < 1")
    this._trainPortion = value
    this
  }

  def setKFold(value: Int): this.type = {
    this._kFold = value
    this._kFoldIteratorRange = Range(0, _kFold).par
    this
  }

  def setSeed(value: Long): this.type = {
    this._seed = value
    this
  }

  def getLabelCol: String = _labelCol
  def getFeaturesCol: String = _featureCol
  def getTrainPortion: Double = _trainPortion
  def getKFold: Int = _kFold
  def getSeed: Long = _seed



}
