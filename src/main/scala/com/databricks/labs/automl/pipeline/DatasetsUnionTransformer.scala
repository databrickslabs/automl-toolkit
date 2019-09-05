package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineUtils
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

class DatasetsUnionTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with HasTransformCalculated {

  final val unionDatasetName = new Param[String](this, "unionDatasetName", "unionDatasetName")

  def setUnionDatasetName(value: String): this.type = set(unionDatasetName, value)

  def getUnionDatasetName: String = $(unionDatasetName)

  def this() = {
    this(Identifiable.randomUID("DatasetsUnionTransformer"))
    setAutomlInternalId(AutoMlPipelineUtils.AUTOML_INTERNAL_ID_COL)
    setTransformCalculated(false)
  }


  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    if(!getTransformCalculated) {
      setTransformCalculated(true)
      return dataset
        .sqlContext
        .sql(s"select * from $getUnionDatasetName")
        .union(dataset.toDF())
    }
    dataset.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): DatasetsUnionTransformer = defaultCopy(extra)

}

object DatasetsUnionTransformer extends DefaultParamsReadable[DatasetsUnionTransformer] {
  override def load(path: String): DatasetsUnionTransformer = super.load(path)
}