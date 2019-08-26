package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.types.{LongType, StructField, StructType}

class AutoMlOutputDatasetTransformer (override val uid: String)
  extends Transformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeaturesColumns {

  def this() = this(Identifiable.randomUID("AutoMlOutputDatasetTransformer"))

  final val tempViewOriginalDatasetName: Param[String] = new Param[String](this, "tempViewOriginalDatasetName", "Temp table name")

  def setTempViewOriginalDatasetName(value: String): this.type = set(tempViewOriginalDatasetName, value)

  def getTempViewOriginalDatasetName: String = $(tempViewOriginalDatasetName)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val originalUserDf =  dataset.sqlContext.sql(s"select * from $getTempViewOriginalDatasetName")
    val userViewDf = dataset
      .drop(getFeatureColumns:_*)
      .join(originalUserDf, AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
      .drop(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
    dataset.sqlContext.dropTempTable(getTempViewOriginalDatasetName)
    userViewDf.toDF()
  }

  override def transformSchema(schema: StructType): StructType = {
    val spark = SparkSession.builder().getOrCreate()
    val originalDfSchema = spark.sql(s"select * from $getTempViewOriginalDatasetName").schema
    StructType(
      schema.fields.filterNot(field => AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL.equals(field.name))
        ++
        originalDfSchema.fields.filterNot(field => getFeatureColumns.contains(field.name)))
  }

  override def copy(extra: ParamMap): AutoMlOutputDatasetTransformer = defaultCopy(extra)

}

object AutoMlOutputDatasetTransformer extends DefaultParamsReadable[AutoMlOutputDatasetTransformer] {

  override def load(path: String): AutoMlOutputDatasetTransformer = super.load(path)

}
