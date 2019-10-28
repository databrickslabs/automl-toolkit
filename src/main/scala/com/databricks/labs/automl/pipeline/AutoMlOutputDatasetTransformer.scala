package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.types.{LongType, StructField, StructType}

/**
  * @author Jas Bali
  * This transformer is intended to be used as a last stage in the inference pipeline.
  * Note: This transformer is supposed to be used with [[ZipRegisterTempTransformer]],
  * as the first transformer in the Inference/Training pipeline. It generates the final
  * dataset that is returned as a result of doing a transform on the [[org.apache.spark.ml.PipelineModel]]
  * This is extremely useful for making sure all the original columns are present in the final
  * transformed dataset, since there may be a need to JOIN operations on ignored columns in the
  * downstream of inference step
  * @param uid
  */
class AutoMlOutputDatasetTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeaturesColumns {

  def this() = {
    this(Identifiable.randomUID("AutoMlOutputDatasetTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setDebugEnabled(false)
  }

  final val tempViewOriginalDatasetName: Param[String] = new Param[String](this, "tempViewOriginalDatasetName", "Temp table name")

  def setTempViewOriginalDatasetName(value: String): this.type = set(tempViewOriginalDatasetName, value)

  def getTempViewOriginalDatasetName: String = $(tempViewOriginalDatasetName)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    val originalUserDf =  dataset.sqlContext.sql(s"select * from $getTempViewOriginalDatasetName")
    val userViewDf =
    if(dataset.columns.contains(getLabelColumn)) {
      val tmpDf = dataset
        .drop(getFeatureColumns:_*)
        .drop(getLabelColumn)
      originalUserDf
        .join(tmpDf, getAutomlInternalId)
        .drop(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    } else {
      val tmpDf = dataset
        .drop(getFeatureColumns:_*)
      originalUserDf
        .join(tmpDf, getAutomlInternalId)
        .drop(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    }
    dataset.sqlContext.dropTempTable(getTempViewOriginalDatasetName)
    userViewDf.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    val spark = SparkSession.builder().getOrCreate()
    if(spark.catalog.tableExists(getTempViewOriginalDatasetName)) {
      val originalDfSchema = spark.sql(s"select * from $getTempViewOriginalDatasetName").schema
      return StructType(
        schema.fields.filterNot(field => AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL.equals(field.name))
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
