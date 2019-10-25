package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

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
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setDebugEnabled(false)
  }

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    val dfs = prepareUnion(
      dataset.sqlContext.sql(s"select * from $getUnionDatasetName"),
      dataset.toDF())
    dfs._1.unionByName(dfs._2)
  }

  private def prepareUnion(df1: DataFrame, df2: DataFrame):  (DataFrame, DataFrame) = {
    validateUnion(df1, df2)
    val colNames = df1.schema.fieldNames
    Sorting.quickSort(colNames)
    val newDf1 = df1.select( colNames map col:_*)
    val newDf2 = df2.select( colNames map col:_*)
    val returnVal = (newDf1, newDf2)
    returnVal
  }

  private def validateUnion(df1: DataFrame, df2: DataFrame): Unit = {
    val df1Cols = df1.schema.fieldNames
    Sorting.quickSort(df1Cols)
    val df2Cols = df2.schema.fieldNames
    Sorting.quickSort(df2Cols)
    val df1SchemaString = df1.select(df1Cols map col:_*).schema.toString()
    val df2SchemaString = df2.select(df2Cols map col:_*).schema.toString()
    assert(df1SchemaString.equals(df2SchemaString),
      s"Different schemas for union DFs. \n DF1 schema $df1SchemaString \n " +
        s"DF2 schema $df2SchemaString \n")
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): DatasetsUnionTransformer = defaultCopy(extra)
}

object DatasetsUnionTransformer extends DefaultParamsReadable[DatasetsUnionTransformer] {
  override def load(path: String): DatasetsUnionTransformer = super.load(path)
}