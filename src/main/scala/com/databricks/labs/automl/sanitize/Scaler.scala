package com.databricks.labs.automl.sanitize

import com.databricks.labs.automl.utils.DataValidation
import org.apache.spark.ml.{Estimator, Model, PipelineStage, Transformer}
import org.apache.spark.ml.feature.{MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler}
import org.apache.spark.sql.DataFrame

class Scaler(df: DataFrame = null) extends DataValidation with SanitizerDefaults {

  private var _featuresCol: String = defaultFeaturesCol

  private var _renamedFeaturesCol: String = defaultRenamedFeaturesCol

  private var _scalerType: String = defaultScalerType

  private var _scalerMin: Double = defaultScalerMin

  private var _scalerMax: Double = defaultScalerMax

  private var _standardScalerMeanFlag: Boolean = defaultStandardScalerMeanFlag

  private var _standardScalerStdDevFlag: Boolean = defaultStandardScalerStdDevFlag

  private var _pNorm: Double = defaultPNorm

  private val _dfFieldNames: Array[String] = if(df != null) df.columns else Array.empty

  private def renameFeaturesCol(): this.type = {
    _renamedFeaturesCol = _featuresCol + "_r"
    this
  }

  def setFeaturesCol(value: String): this.type = {
    if(_dfFieldNames.nonEmpty) {
      require(_dfFieldNames.contains(value), s"Feature Column '$value' is not present in Dataframe schema.")
    }
    _featuresCol = value
    renameFeaturesCol()
    this
  }

  def setScalerType(value: String): this.type = {
    require(allowableScalers.contains(value), s"Scaler Type '$value' is not a valid member of ${
      invalidateSelection(value, allowableScalers)
    }")
    _scalerType = value
    this
  }

  def setScalerMin(value: Double): this.type = {
    _scalerMin = value
    this
  }

  def setScalerMax(value: Double): this.type = {
    _scalerMax = value
    this
  }

  def setStandardScalerMeanMode(value: Boolean): this.type = {
    _standardScalerMeanFlag = value
    this
  }

  def setStandardScalerStdDevMode(value: Boolean): this.type = {
    _standardScalerStdDevFlag = value
    this
  }

  def setPNorm(value: Double): this.type = {
    require(value >= 1.0, s"pNorm value must be greater than or equal to 1.0.  '$value' is invalid.")
    _pNorm = value
    this
  }

  def getFeaturesCol: String = _featuresCol

  def getScalerType: String = _scalerType

  def getScalerMin: Double = _scalerMin

  def getScalerMax: Double = _scalerMax

  def getStandardScalerMeanFlag: Boolean = _standardScalerMeanFlag

  def getStandardScalerStdDevFlag: Boolean = _standardScalerStdDevFlag

  def getAllowableScalers: Array[String] = allowableScalers

  def getPNorm: Double = _pNorm

  private def normalizeFeatures(): DataFrame = {

    val normalizer = normalizeFeaturesStage()

    normalizer.transform(
      df.withColumnRenamed(_featuresCol, _renamedFeaturesCol)
    )
      .drop(_renamedFeaturesCol)

  }

  private def normalizeFeaturesStage() : Transformer = {
    new Normalizer()
      .setInputCol(_renamedFeaturesCol)
      .setOutputCol(_featuresCol)
      .setP(_pNorm)
  }

  private def minMaxFeatures(): DataFrame = {

    require(_scalerMax > _scalerMin, s"Scaler Max (${_scalerMax}) must be greater than Scaler Min (${_scalerMin})")
    val minMaxScaler = minMaxFeaturesStage()

    val dfRenamed = df.withColumnRenamed(_featuresCol, _renamedFeaturesCol)

    val fitScaler = minMaxScaler.fit(dfRenamed)

    fitScaler.transform(dfRenamed).drop(_renamedFeaturesCol)
  }

  private def minMaxFeaturesStage(): Estimator[_ <: Model[_]] = {
    new MinMaxScaler()
      .setInputCol(_renamedFeaturesCol)
      .setOutputCol(_featuresCol)
      .setMin(_scalerMin)
      .setMax(_scalerMax)
  }

  private def standardScaleFeatures(): DataFrame = {

    val standardScaler = standardScaleFeaturesStage()

    val dfRenamed = df.withColumnRenamed(_featuresCol, _renamedFeaturesCol)

    val fitStandardScaler = standardScaler.fit(dfRenamed)

    fitStandardScaler.transform(dfRenamed).drop(_renamedFeaturesCol)

  }

  private def standardScaleFeaturesStage(): Estimator[_ <: Model[_]] = {
    new StandardScaler()
      .setInputCol(_renamedFeaturesCol)
      .setOutputCol(_featuresCol)
      .setWithMean(_standardScalerMeanFlag)
      .setWithStd(_standardScalerStdDevFlag)
  }

  private def maxAbsScaleFeatures(): DataFrame = {

    val maxAbsScaler = maxAbsScaleFeaturesStage()

    val dfRenamed = df.withColumnRenamed(_featuresCol, _renamedFeaturesCol)

    val fitMaxAbsScaler = maxAbsScaler.fit(dfRenamed)

    fitMaxAbsScaler.transform(dfRenamed).drop(_renamedFeaturesCol)

  }

  private def maxAbsScaleFeaturesStage(): Estimator[_ <: Model[_]] = {
    new MaxAbsScaler()
      .setInputCol(_renamedFeaturesCol)
      .setOutputCol(_featuresCol)
  }

  def scaleFeatures(): DataFrame = {
    _scalerType match {
      case "minMax" => minMaxFeatures()
      case "standard" => standardScaleFeatures()
      case "normalize" => normalizeFeatures()
      case "maxAbs" => maxAbsScaleFeatures()
      case _ => throw new UnsupportedOperationException(s"Scaler '${_scalerType}' is not supported.")
    }
  }


  def scaleFeaturesForPipeline(): PipelineStage = {
    _scalerType match {
      case "minMax" => minMaxFeaturesStage()
      case "standard" => standardScaleFeaturesStage()
      case "normalize" => normalizeFeaturesStage()
      case "maxAbs" => maxAbsScaleFeaturesStage()
      case _ => throw new UnsupportedOperationException(s"Scaler '${_scalerType}' is not supported.")
    }
  }

}
