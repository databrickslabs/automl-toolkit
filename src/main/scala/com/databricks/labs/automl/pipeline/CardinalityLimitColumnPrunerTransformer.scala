package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.SchemaUtils
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

class CardinalityLimitColumnPrunerTransformer(override val uid: String)
  extends Transformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasTransformCalculated {

  def this() = this(Identifiable.randomUID("CardinalityLimitColumnPrunerTransformer"))

  final val cardinalityLimit: IntParam = new IntParam(
    this, "cardinalityLimit",
    "Setting this to a limit will ignore columns with cardinality higher than this limit")

  final val prunedColumns: StringArrayParam = new StringArrayParam(this, "prunedColumns", "Columns to ignore based on cardinality limit")

  def setCardinalityLimit(value: Int): this.type = set(cardinalityLimit, value)

  def getCardinalityLimit: Int = $(cardinalityLimit)

  def setPrunedColumns(value: Array[String]): this.type = set(prunedColumns, value)

  def getPrunedColumns: Array[String] = $(prunedColumns)


  override def transform(dataset: Dataset[_]): DataFrame = {
    if(!getTransformCalculated) {
      val columnTypes = SchemaUtils.extractTypes(dataset.toDF(), getLabelColumn)
      if(SchemaUtils.isNotEmpty(columnTypes._2)) {
        val columnsToDrop = SchemaUtils.validateCardinality(dataset.toDF(), columnTypes._2).invalidFields
        if(SchemaUtils.isEmpty(getPrunedColumns)) {
          setPrunedColumns(columnsToDrop.toArray[String])
        }
        transformSchema(dataset.schema)
        setTransformCalculated(true)
        return dataset.drop(columnsToDrop:_*).toDF()
      }
    }
    transformSchema(dataset.schema)
    if(SchemaUtils.isNotEmpty(getPrunedColumns.toList)) {
      return dataset.drop(getPrunedColumns:_*).toDF()
    }
    dataset.toDF()
  }

  override def transformSchema(schema: StructType): StructType = {
    if(SchemaUtils.isNotEmpty(getPrunedColumns.toList)) {
      val allCols = schema.fields.map(field => field.name)
      val missingCols = getPrunedColumns.filterNot(colName => allCols.contains(colName))
      if(missingCols.nonEmpty) {
        throw new RuntimeException(s"""Following columns are missing: ${missingCols.mkString(", ")}""")
      }
      return StructType(schema.fields.filterNot(field => getPrunedColumns.contains(field.name)))
    }
    schema
  }

  override def copy(extra: ParamMap): CardinalityLimitColumnPrunerTransformer = defaultCopy(extra)

}

object CardinalityLimitColumnPrunerTransformer extends DefaultParamsReadable[CardinalityLimitColumnPrunerTransformer] {

  override def load(path: String): CardinalityLimitColumnPrunerTransformer = super.load(path)

}

