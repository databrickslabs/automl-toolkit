package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.types.{LongType, StructField, StructType}

class AutoMlOutputDatasetTransformer (override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeaturesColumns {

  def this() = {
    this(Identifiable.randomUID("AutoMlOutputDatasetTransformer"))
    setAutomlInternalId(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
  }

  final val tempViewOriginalDatasetName: Param[String] = new Param[String](this, "tempViewOriginalDatasetName", "Temp table name")

  def setTempViewOriginalDatasetName(value: String): this.type = set(tempViewOriginalDatasetName, value)

  def getTempViewOriginalDatasetName: String = $(tempViewOriginalDatasetName)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    val originalUserDf =  dataset.sqlContext.sql(s"select * from $getTempViewOriginalDatasetName")
    val userViewDf =
    if(dataset.columns.contains(getLabelColumn)) {
      dataset
        .drop(getFeatureColumns:_*)
        .drop(getLabelColumn)
        .join(originalUserDf, getAutomlInternalId)
        .drop(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
    } else {
      dataset
        .drop(getFeatureColumns:_*)
        .join(originalUserDf, getAutomlInternalId)
        .drop(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
    }
    dataset.sqlContext.dropTempTable(getTempViewOriginalDatasetName)
    userViewDf.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    val spark = SparkSession.builder().getOrCreate()
    if(spark.catalog.tableExists(getTempViewOriginalDatasetName)) {
      val originalDfSchema = spark.sql(s"select * from $getTempViewOriginalDatasetName").schema
      return StructType(
        schema.fields.filterNot(field => AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL.equals(field.name))
          ++
          originalDfSchema.fields.filterNot(field => getFeatureColumns.contains(field.name)))
    }
    schema
  }

  override def copy(extra: ParamMap): AutoMlOutputDatasetTransformer = defaultCopy(extra)

}

object AutoMlOutputDatasetTransformer extends DefaultParamsReadable[AutoMlOutputDatasetTransformer] {

  override def load(path: String): AutoMlOutputDatasetTransformer = super.load(path)

}
