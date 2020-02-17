package com.databricks.labs.automl.pipeline

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCols}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * @author Jas Bali
  * This is a useful transformer, if there is a need to rename columns
  * in the intermediate transformations of a pipeline. Using this transformer
  * can help avoid doing intermediate "fit" on pipeline just to rename columns
  * in the output dataset
  *
  * Note: This is a noops transformer if input columns are not present in the dataset
  */
class ColumnNameTransformer(override val uid: String)
  extends Transformer
  with DefaultParamsWritable
  with HasInputCols
  with HasOutputCols
  with HasDebug
  with HasPipelineId {

  def this() = {
    this(Identifiable.randomUID("ColumnNameTransformer"))
    setDebugEnabled(false)
  }

  def setInputColumns(value: Array[String]): this.type = set(inputCols, value)

  def setOutputColumns(value: Array[String]): this.type = set(outputCols, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val startMillis = System.currentTimeMillis()
    if(getInputCols.forall(item => dataset.columns.contains(item))) {
      transformSchema(dataset.schema)
      var newDataset = dataset
      for((key, i) <- getInputCols.view.zipWithIndex) {
        newDataset = dataset.withColumnRenamed(key, getOutputCols(i))
      }
      logTransformation(dataset, newDataset, System.currentTimeMillis() - startMillis)
      return newDataset.toDF()
    }
    dataset.toDF()
  }

  override def transformSchema(schema: StructType): StructType = {
    require(
     getInputCols.length == getOutputCols.length,
     s"${getInputCols.toList} input columns array is not equal in length to output columns array ${getOutputCols.toList}")
    StructType(schema.fields.zipWithIndex.map{case (element, index) =>
      if(getInputCols.contains(element.name)) {
        StructField(getOutputCols(getInputCols.indexOf(element.name)), element.dataType, element.nullable, element.metadata)
      } else {
        element
      }
    })
  }

  override def copy(extra: ParamMap): ColumnNameTransformer = defaultCopy(extra)
}

object ColumnNameTransformer extends DefaultParamsReadable[ColumnNameTransformer] {
  override def load(path: String): ColumnNameTransformer = super.load(path)
}
