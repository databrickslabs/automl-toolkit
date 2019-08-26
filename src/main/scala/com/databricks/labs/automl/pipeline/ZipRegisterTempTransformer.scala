package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

class ZipRegisterTempTransformer(override val uid: String)
  extends Transformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeaturesColumns {

  def this() = this(Identifiable.randomUID("ZipRegisterTempTransformer"))

  final val tempViewOriginalDatasetName: Param[String] = new Param[String](this, "tempViewOriginalDatasetName", "Temp table name")

  def setTempViewOriginalDatasetName(value: String): this.type = set(tempViewOriginalDatasetName, value)

  def getTempViewOriginalDatasetName: String = $(tempViewOriginalDatasetName)


  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val dfWithId = dataset
      .withColumn(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL, monotonically_increasing_id())

    dfWithId.createOrReplaceTempView(getTempViewOriginalDatasetName)

    val colsToSelect =
      (Array(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL, getLabelColumn) ++ getFeatureColumns)
      .map(field => col(field))

    dfWithId.select(colsToSelect:_*)
  }

  override def transformSchema(schema: StructType): StructType = {
    StructType(schema.fields.filter(field => $(featureColumns).contains(field.name))
      :+
      StructField(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL, LongType, nullable = false)
      :+
      StructField(getLabelColumn, StringType, nullable=false))
  }

  override def copy(extra: ParamMap): ZipRegisterTempTransformer = defaultCopy(extra)

}

object ZipRegisterTempTransformer extends DefaultParamsReadable[ZipRegisterTempTransformer] {

  override def load(path: String): ZipRegisterTempTransformer = super.load(path)

}
