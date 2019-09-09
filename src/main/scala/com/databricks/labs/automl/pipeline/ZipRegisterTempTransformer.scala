package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

class ZipRegisterTempTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeaturesColumns {

  def this() = {
    this(Identifiable.randomUID("ZipRegisterTempTransformer"))
    setAutomlInternalId(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
  }

  final val tempViewOriginalDatasetName: Param[String] = new Param[String](this, "tempViewOriginalDatasetName", "Temp table name")

  def setTempViewOriginalDatasetName(value: String): this.type = set(tempViewOriginalDatasetName, value)

  def getTempViewOriginalDatasetName: String = $(tempViewOriginalDatasetName)


  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    val dfWithId = dataset
      .withColumn(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL, monotonically_increasing_id())

    dfWithId.createOrReplaceTempView(getTempViewOriginalDatasetName)

    val colsSelectTmp = if(dfWithId.columns.contains(getLabelColumn)) {
      Array(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL, getLabelColumn)
    } else {
      Array(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
    }

    val colsToSelect =
      (colsSelectTmp ++ getFeatureColumns)
      .map(field => col(field))

    dfWithId.select(colsToSelect:_*)
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
   if(schema.fieldNames.contains(getLabelColumn)) {
      StructType(schema.fields.filter(field => $(featureColumns).contains(field.name))
        :+
        StructField(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL, LongType, nullable = false)
        :+
        StructField(getLabelColumn, StringType, nullable=false))
    } else {
      StructType(schema.fields.filter(field => $(featureColumns).contains(field.name))
        :+
        StructField(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL, LongType, nullable = false))
    }
  }

  override def copy(extra: ParamMap): ZipRegisterTempTransformer = defaultCopy(extra)

}

object ZipRegisterTempTransformer extends DefaultParamsReadable[ZipRegisterTempTransformer] {

  override def load(path: String): ZipRegisterTempTransformer = super.load(path)

}
