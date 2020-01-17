package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.inference.NaFillConfig
import com.databricks.labs.automl.sanitize.DataSanitizer
import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, SchemaUtils}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsWritable,
  Identifiable
}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * @author Jas Bali
  * Input: Original feature columns
  * Output: sanitized, nafilled columns
  *
  */
class DataSanitizerTransformer(override val uid: String)
    extends AbstractTransformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeatureColumn {

  final val numericFillStat: Param[String] =
    new Param[String](this, "numericFillStat", "Numeric fill stats")
  final val characterFillStat: Param[String] =
    new Param[String](this, "characterFillStat", "Character fill stat")
  final val modelSelectionDistinctThreshold: IntParam = new IntParam(
    this,
    "modelSelectionDistinctThreshold",
    "model selection distinct threshold"
  )
  final val filterPrecision: DoubleParam =
    new DoubleParam(this, "filterPrecision", "Filter precision")
  final val parallelism: IntParam =
    new IntParam(this, "parallelism", "filter parallelism")
  final val naFillFlag: BooleanParam =
    new BooleanParam(this, "naFillFlag", "Na Fill flag")
  final val categoricalColumnNames =
    new StringArrayParam(this, "categoricalColumnNames", "Categorical Columns")
  final val categoricalColumnValues = new StringArrayParam(
    this,
    "categoricalColumnValues",
    "Categorical Columns' Values"
  )
  final val numericColumnNames =
    new StringArrayParam(this, "numericColumnNames", "Numeric Columns")
  final val numericColumnValues =
    new DoubleArrayParam(this, "numericColumnValues", "Numeric Columns' Values")
  final val booleanColumnNames =
    new StringArrayParam(this, "booleanColumnNames", "Boolean Columns")
  final val booleanColumnValues =
    new StringArrayParam(this, "booleanColumnValues", "Boolean Columns' Values")
  final val decideModel: Param[String] =
    new Param[String](this, "decideModel", "Decided model")
  final val fillMode: Param[String] =
    new Param[String](this, "fillMode", "fillMode")
  final val characterNABlanketFill: Param[String] =
    new Param[String](this, "characterNABlanketFill", "characterNABlanketFill")
  final val numericNABlanketFill: DoubleParam =
    new DoubleParam(this, "numericNABlanketFill", "numericNABlanketFill")
  final val categoricalNAFillMapKeys: StringArrayParam = new StringArrayParam(
    this,
    "categoricalNAFillMapKeys",
    "categoricalNAFillMapKeys"
  )
  final val categoricalNAFillMapValues: StringArrayParam = new StringArrayParam(
    this,
    "categoricalNAFillMapValues",
    "categoricalNAFillMapValues"
  )
  final val numericNAFillMapKeys: StringArrayParam =
    new StringArrayParam(this, "numericNAFillMapKeys", "numericNAFillMapKeys")
  final val numericNAFillMapValues: DoubleArrayParam = new DoubleArrayParam(
    this,
    "numericNAFillMapValues",
    "numericNAFillMapValues"
  )

  def setNumericFillStat(value: String): this.type = set(numericFillStat, value)

  def getNumericFillStat: String = $(numericFillStat)

  def setCharacterFillStat(value: String): this.type =
    set(characterFillStat, value)

  def getCharacterFillStat: String = $(characterFillStat)

  def setModelSelectionDistinctThreshold(value: Int): this.type =
    set(modelSelectionDistinctThreshold, value)

  def getModelSelectionDistinctThreshold: Int =
    $(modelSelectionDistinctThreshold)

  def setFilterPrecision(value: Double): this.type = set(filterPrecision, value)

  def getFilterPrecision: Double = $(filterPrecision)

  def setParallelism(value: Int): this.type = set(parallelism, value)

  def getParallelism: Int = $(parallelism)

  def setNaFillFlag(value: Boolean): this.type = set(naFillFlag, value)

  def getNaFillFlag: Boolean = $(naFillFlag)

  def setCategoricalColumnNames(value: Array[String]): this.type =
    set(categoricalColumnNames, value)

  def getCategoricalColumnNames: Array[String] = $(categoricalColumnNames)

  def setCategoricalColumnValues(value: Array[String]): this.type =
    set(categoricalColumnValues, value)

  def getCategoricalColumnValues: Array[String] = $(categoricalColumnValues)

  def setNumericColumnNames(value: Array[String]): this.type =
    set(numericColumnNames, value)

  def getNumericColumnNames: Array[String] = $(numericColumnNames)

  def setBooleanColumnNames(value: Array[String]): this.type =
    set(booleanColumnNames, value)

  def getBooleanColumnNames: Array[String] = $(booleanColumnNames)

  def setBooleanColumnValues(value: Array[Boolean]): this.type =
    set(booleanColumnValues, value.map(_.toString))

  def getBooleanColumnValues: Array[Boolean] =
    $(booleanColumnValues).map(_.toBoolean)

  def setNumericColumnValues(value: Array[Double]): this.type =
    set(numericColumnValues, value)

  def getNumericColumnValues: Array[Double] = $(numericColumnValues)

  def setDecideModel(value: String): this.type = set(decideModel, value)

  def getDecideModel: String = $(decideModel)

  def setFillMode(value: String): this.type = set(fillMode, value)

  def getFillMode: String = $(fillMode)

  def setCharacterNABlanketFill(value: String): this.type =
    set(characterNABlanketFill, value)

  def getCharacterNABlanketFill: String = $(characterNABlanketFill)

  def setNumericNABlanketFill(value: Double): this.type =
    set(numericNABlanketFill, value)

  def getNumericNABlanketFill: Double = $(numericNABlanketFill)

  def setCategoricalNAFillMapKeys(value: Array[String]): this.type =
    set(categoricalNAFillMapKeys, value)

  def getCategoricalNAFillMapKeys: Array[String] = $(categoricalNAFillMapKeys)

  def setCategoricalNAFillMapValues(value: Array[String]): this.type =
    set(categoricalNAFillMapValues, value)

  def getCategoricalNAFillMapValues: Array[String] =
    $(categoricalNAFillMapValues)

  def setNumericNAFillMapKeys(value: Array[String]): this.type =
    set(numericNAFillMapKeys, value)

  def getNumericNAFillMapKeys: Array[String] = $(numericNAFillMapKeys)

  def setNumericNAFillMapValues(value: Array[Double]): this.type =
    set(numericNAFillMapValues, value)

  def getNumericNAFillMapValues: Array[Double] = $(numericNAFillMapValues)

  def setCategoricalNAFillMap(value: Map[String, String]): this.type = {
    setCategoricalNAFillMapKeys(value.keys.toArray)
    setCategoricalNAFillMapValues(value.values.toArray)
  }

  def setNumericNAFillMap(value: Map[String, Double]): this.type = {
    setNumericNAFillMapKeys(value.keys.toArray)
    setNumericNAFillMapValues(value.values.toArray)
  }

  def this() = {
    this(Identifiable.randomUID("DataSanitizerTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setFeatureCol("features")
    setNumericFillStat("mean")
    setCharacterFillStat("max")
    setModelSelectionDistinctThreshold(10)
    setFilterPrecision(0.01)
    setParallelism(20)
    setNaFillFlag(false)
    setDecideModel("")
    setCategoricalColumnNames(Array.empty)
    setNumericColumnValues(Array.empty)
    setNumericColumnNames(Array.empty)
    setNumericColumnValues(Array.empty)
    setBooleanColumnNames(Array.empty)
    setBooleanColumnValues(Array.empty)
    setCategoricalNAFillMapKeys(Array.empty)
    setCategoricalNAFillMapValues(Array.empty)
    setNumericNAFillMapKeys(Array.empty)
    setNumericNAFillMapValues(Array.empty)
    setCharacterNABlanketFill("")
    setNumericNABlanketFill(0.0)
    setFillMode("auto")
    setDebugEnabled(false)
  }

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    val naConfig = new DataSanitizer(dataset.toDF())
      .setLabelCol(getLabelColumn)
      .setFeatureCol(getFeatureCol)
      .setModelSelectionDistinctThreshold(getModelSelectionDistinctThreshold)
      .setNumericFillStat(getNumericFillStat)
      .setCharacterFillStat(getCharacterFillStat)
      .setParallelism(getParallelism)
      .setCategoricalNAFillMap(
        SchemaUtils.generateMapFromKeysValues(
          getCategoricalNAFillMapKeys,
          getCategoricalNAFillMapValues
        )
      )
      .setCharacterNABlanketFillValue(getCharacterNABlanketFill)
      .setNumericNABlanketFillValue(getNumericNABlanketFill)
      .setNumericNAFillMap(
        SchemaUtils.generateMapFromKeysValues(
          getNumericNAFillMapKeys,
          getNumericNAFillMapValues
        )
      )
      .setNAFillMode(getFillMode)
      .setFilterPrecision(getFilterPrecision)
      .setFieldsToIgnoreInVector(Array(getAutomlInternalId))

    val (naFilledDataFrame, fillMap, detectedModelType) =
      if (getNaFillFlag) {
        val naFillConfigTmp = buildNaConfig()
        if (naFillConfigTmp.isDefined) {
          naConfig.generateCleanData(
            naFillConfigTmp.get,
            refactorLabelFlag = false,
            decidedModel = getDecideModel
          )
        } else {
          naConfig.generateCleanData(
            refactorLabelFlag = false,
            decidedModel = getDecideModel
          )
        }
      } else {
        (
          dataset,
          NaFillConfig(Map("" -> ""), Map("" -> 0.0), Map("" -> false)),
          naConfig.decideModel()
        )
      }
    if (getDecideModel == null || getDecideModel.isEmpty) {
      setCategoricalColumnNames(fillMap.categoricalColumns.keys.toArray)
      setCategoricalColumnValues(fillMap.categoricalColumns.values.toArray)
      setNumericColumnNames(fillMap.numericColumns.keys.toArray)
      setNumericColumnValues(fillMap.numericColumns.values.toArray)
      setDecideModel(detectedModelType)
    }

    naFilledDataFrame.toDF()
  }

  private def buildNaConfig(): Option[NaFillConfig] = {
    if (SchemaUtils.isNotEmpty(getCategoricalColumnNames) &&
        SchemaUtils.isNotEmpty(getNumericColumnNames)) {
      return Some(
        NaFillConfig(
          categoricalColumns = SchemaUtils.generateMapFromKeysValues(
            getCategoricalColumnNames,
            getCategoricalColumnValues
          ),
          numericColumns = SchemaUtils.generateMapFromKeysValues(
            getNumericColumnNames,
            getNumericColumnValues
          ),
          booleanColumns = SchemaUtils.generateMapFromKeysValues(
            getBooleanColumnNames,
            getBooleanColumnValues
          )
        )
      )
    }
    None
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): DataSanitizerTransformer =
    defaultCopy(extra)
}

object DataSanitizerTransformer
    extends DefaultParamsReadable[DataSanitizerTransformer] {
  override def load(path: String): DataSanitizerTransformer = super.load(path)
}
