package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * @author Jas Bali
  * This transformer stage is supposed to be the first stage of a pipeline and is useful for adding an
  * ID column to the input dataset, drop ignored columns and register original dataset a temp. view
  * This is supposed to work with [[AutoMlOutputDatasetTransformer]] as the last stage of a pipeline
  * which reverts the transformed dataset with the ignored fields using ID column
  * @param uid
  */
class ZipRegisterTempTransformer(override val uid: String)
    extends AbstractTransformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeaturesColumns {

  def this() = {
    this(Identifiable.randomUID("ZipRegisterTempTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setDebugEnabled(false)
  }

  final val tempViewOriginalDatasetName: Param[String] =
    new Param[String](this, "tempViewOriginalDatasetName", "Temp table name")

  def setTempViewOriginalDatasetName(value: String): this.type =
    set(tempViewOriginalDatasetName, value)

  final val dataPrepCachingFlag: BooleanParam =
    new BooleanParam(this, "dataPrepCachingFlag", "data prep stage caching flag")

  def getTempViewOriginalDatasetName: String = $(tempViewOriginalDatasetName)

  def setDataPrepCachingFlag(value: Boolean): this.type = set(dataPrepCachingFlag, value)

  def getDataPrepCachingFlag: Boolean = $(dataPrepCachingFlag)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    val dfWithId = dataset
      .filter(col($(labelColumn)).isNotNull)
      .filter(!col($(labelColumn)).isNaN)
      .withColumn(
        AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL,
        monotonically_increasing_id()
      )
    dfWithId.createOrReplaceTempView(getTempViewOriginalDatasetName)
    val colsSelectTmp = if (dfWithId.columns.contains(getLabelColumn)) {
      Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL, getLabelColumn)
    } else {
      Array(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    }
    val colsToSelect =
      (colsSelectTmp ++ getFeatureColumns)
        .map(field => col(field))
    dfWithId.select(colsToSelect: _*)
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    if (schema.fieldNames.contains(getLabelColumn)) {
      StructType(
        schema.fields.filter(field => $(featureColumns).contains(field.name))
          :+
            StructField(
              AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL,
              LongType,
              nullable = false
            )
          :+
            StructField(getLabelColumn, StringType, nullable = false)
      )
    } else {
      StructType(
        schema.fields.filter(field => $(featureColumns).contains(field.name))
          :+
            StructField(
              AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL,
              LongType,
              nullable = false
            )
      )
    }
  }

  override def copy(extra: ParamMap): ZipRegisterTempTransformer =
    defaultCopy(extra)
}

object ZipRegisterTempTransformer
    extends DefaultParamsReadable[ZipRegisterTempTransformer] {
  override def load(path: String): ZipRegisterTempTransformer = super.load(path)
}
