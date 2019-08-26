package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.inference.NaFillConfig
import com.databricks.labs.automl.sanitize.DataSanitizer
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}


class DataSanitizerTransformer(override val uid: String)
  extends Transformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeatureColumn {

  final val numericFillStat: Param[String] = new Param[String](this, "numericFillStat", "Numeric fill stats")
  final val characterFillStat: Param[String] = new Param[String](this, "characterFillStat", "Character fill stat")
  final val modelSelectionDistinctThreshold: IntParam = new IntParam(this, "modelSelectionDistinctThreshold", "model selection distinct threshold")
  final val filterPrecision: DoubleParam = new DoubleParam(this, "filterPrecision", "Filter precision")
  final val parallelism: IntParam = new IntParam(this, "parallelism", "filter parallelism")
  final val naFillFlag: BooleanParam = new BooleanParam(this, "naFillFlag", "Na Fill flag")
  final val categoricalColumns = new Param[Map[String, String]](this, "categoricalColumns", "Categorical Columns")
  final val numericColumns = new Param[Map[String, Double]](this, "numericColumns", "Numeric Columns")
  final val decideModel: Param[String] = new Param[String](this, "decideModel", "Decided model")

  def setNumericFillStat(value: String): this.type = set(numericFillStat, value)

  def getNumericFillStat: String = $(numericFillStat)

  def setCharacterFillStat(value: String): this.type = set(characterFillStat, value)

  def getCharacterFillStat: String = $(characterFillStat)

  def setModelSelectionDistinctThreshold(value: Int): this.type = set(modelSelectionDistinctThreshold, value)

  def getModelSelectionDistinctThreshold: Int = $(modelSelectionDistinctThreshold)

  def setFilterPrecision(value: Double): this.type = set(filterPrecision, value)

  def getFilterPrecision: Double = $(filterPrecision)

  def setParallelism(value: Int): this.type = set(parallelism, value)

  def getParallelism:Int = $(parallelism)

  def setNaFillFlag(value: Boolean): this.type = set(naFillFlag, value)

  def getNaFillFlag: Boolean = $(naFillFlag)

  def setCategoricalColumns(value: Map[String, String]): this.type = set(categoricalColumns, value)

  def getCategoricalColumns: Map[String, String] = $(categoricalColumns)

  def setNumericColumns(value: Map[String, Double]): this.type = set(numericColumns, value)

  def getNumericColumns: Map[String, Double] = $(numericColumns)

  def setDecideModel(value: String): this.type = set(decideModel, value)

  def getDecideModel: String = $(decideModel)


  def this() = {
    this(Identifiable.randomUID("DataSanitizerTransformer"))
    setFeatureCol("features")
    setNumericFillStat("mean")
    setCharacterFillStat("max")
    setModelSelectionDistinctThreshold(10)
    setFilterPrecision(0.01)
    setParallelism(20)
    setNaFillFlag(false)
  }


  override def transform(dataset: Dataset[_]): DataFrame = {
    val naConfig = new DataSanitizer(dataset.toDF())
      .setLabelCol(getLabelColumn)
      .setFeatureCol(getFeatureCol)
      .setModelSelectionDistinctThreshold(getModelSelectionDistinctThreshold)
      .setNumericFillStat(getNumericFillStat)
      .setCharacterFillStat(getCharacterFillStat)
      .setParallelism(getParallelism)

    val (naFilledDataFrame, fillMap, detectedModelType) =
      if (getNaFillFlag) {
        naConfig.generateCleanData()
      } else {
        (
          dataset,
          NaFillConfig(Map("" -> ""), Map("" -> 0.0)),
          naConfig.decideModel()
        )
      }
    if(getDecideModel == null || getDecideModel.isEmpty) {
      setCategoricalColumns(fillMap.categoricalColumns)
      setNumericColumns(fillMap.numericColumns)
      setDecideModel(detectedModelType)
    }

    naFilledDataFrame.toDF()
  }

  override def transformSchema(schema: StructType): StructType = {
   schema
  }

  override def copy(extra: ParamMap): DataSanitizerTransformer = defaultCopy(extra)

}

object DataSanitizerTransformer extends DefaultParamsReadable[DataSanitizerTransformer] {

  override def load(path: String): DataSanitizerTransformer = super.load(path)

}
