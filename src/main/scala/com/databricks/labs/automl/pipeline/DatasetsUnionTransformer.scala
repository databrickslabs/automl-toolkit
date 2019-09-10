package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.sanitize.FeatureCorrelationDetection
import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._

import scala.util.Sorting

/**
  * @author Jas Bali
  * A transformer stage that is useful to do joins on two datasets. It is useful
  * when there is a need to do a join on two datasets in the intermediate step of a pipeline
  *
  * NOTE: A transformer semantics does not allow to pass two datasets to a transform method.
  * As a workaround, the first dataset needs to be registered as a temp table outside of this transformer
  * using [[RegisterTempTableTransformer]] transformer.
  */
class DatasetsUnionTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable {

  final val unionDatasetName = new Param[String](this, "unionDatasetName", "unionDatasetName")

  def setUnionDatasetName(value: String): this.type = set(unionDatasetName, value)

  def getUnionDatasetName: String = $(unionDatasetName)

  def this() = {
    this(Identifiable.randomUID("DatasetsUnionTransformer"))
    setAutomlInternalId(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
    setDebugEnabled(false)
  }


  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    val dfs = prepareUnion(
      dataset.sqlContext.sql(s"select * from $getUnionDatasetName"),
      dataset.toDF())
    dfs._1.union(dfs._2)
  }

  private def prepareUnion(df1: DataFrame, df2: DataFrame):  (DataFrame, DataFrame) = {
    assert(df1.columns.length == df2.columns.length, "Different number of columns")
    val colNames = df1.schema.fieldNames
    Sorting.quickSort(colNames)
    (df1.select( colNames map col:_*), df2.select( colNames map col:_*))
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): DatasetsUnionTransformer = defaultCopy(extra)

}

object DatasetsUnionTransformer extends DefaultParamsReadable[DatasetsUnionTransformer] {
  override def load(path: String): DatasetsUnionTransformer = super.load(path)
}