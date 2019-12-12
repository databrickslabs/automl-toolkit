package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.utils.data.CategoricalHandler
import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, SchemaUtils}
import org.apache.spark.ml.param.{
  DoubleParam,
  IntParam,
  Param,
  ParamMap,
  StringArrayParam
}
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsWritable,
  Identifiable
}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * @author Jas Bali
  * A transformer to apply cardinality limit rules to the input dataset.
  * Given a cardinality limit, this transformer will drop all columns with
  * the cardinality higher than that of a pre-defined limit
  */
class CardinalityLimitColumnPrunerTransformer(override val uid: String)
    extends AbstractTransformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasTransformCalculated {

  def this() = {
    this(Identifiable.randomUID("CardinalityLimitColumnPrunerTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setCardinalityLimit(500)
    setTransformCalculated(false)
    setPrunedColumns(null)
    setDebugEnabled(false)
  }

  final val cardinalityLimit: IntParam = new IntParam(
    this,
    "cardinalityLimit",
    "Setting this to a limit will ignore columns with cardinality higher than this limit"
  )

  final val cardinalityCheckMode: Param[String] =
    new Param[String](this, "cardinalityCheckMode", "cardinality chec mode")

  final val cardinalityType: Param[String] =
    new Param[String](this, "cardinalityType", "cardinality type")

  final val cardinalityPrecision: DoubleParam =
    new DoubleParam(this, "cardinalityPrecision", "cardinality precision")

  final val prunedColumns: StringArrayParam = new StringArrayParam(
    this,
    "prunedColumns",
    "Columns to ignore based on cardinality limit"
  )

  def setCardinalityLimit(value: Int): this.type = set(cardinalityLimit, value)

  def getCardinalityLimit: Int = $(cardinalityLimit)

  def setPrunedColumns(value: Array[String]): this.type =
    set(prunedColumns, value)

  def getPrunedColumns: Array[String] = $(prunedColumns)

  def setCardinalityCheckMode(value: String): this.type =
    set(cardinalityCheckMode, value)

  def getCardinalityCheckMode: String = $(cardinalityCheckMode)

  def setCardinalityType(value: String): this.type = set(cardinalityType, value)

  def getCardinalityType: String = $(cardinalityType)

  def setCardinalityPrecision(value: Double): this.type =
    set(cardinalityPrecision, value)

  def getCardinalityPrecision: Double = $(cardinalityPrecision)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    if (!getTransformCalculated) {
      val columnTypes = SchemaUtils.extractTypes(dataset.toDF(), getLabelColumn)
      if (SchemaUtils.isNotEmpty(columnTypes.categoricalFields)) {

        val colsValidated =
          new CategoricalHandler(dataset.toDF(), getCardinalityCheckMode)
            .setCardinalityType(getCardinalityType)
            .setPrecision(getCardinalityPrecision)
            .validateCategoricalFields(
              columnTypes.categoricalFields
                .filterNot(item => getAutomlInternalId.equals(item)),
              getCardinalityLimit
            )

        val columnsToDrop = columnTypes.categoricalFields
          .filterNot(col => colsValidated.contains(col))

        if (SchemaUtils.isEmpty(getPrunedColumns)) {
          setPrunedColumns(columnsToDrop.toArray[String])
        }
        setTransformCalculated(true)
        return dataset.drop(columnsToDrop: _*).toDF()
      }
    }
    if (SchemaUtils.isNotEmpty(getPrunedColumns)) {
      return dataset.drop(getPrunedColumns: _*).toDF()
    }
    dataset.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    if (SchemaUtils.isNotEmpty(getPrunedColumns)) {
      val allCols = schema.fields.map(field => field.name)
      val missingCols =
        getPrunedColumns.filterNot(colName => allCols.contains(colName))
      if (missingCols.nonEmpty) {
        throw new RuntimeException(
          s"""Following columns are missing: ${missingCols.mkString(", ")}"""
        )
      }
      return StructType(
        schema.fields.filterNot(field => getPrunedColumns.contains(field.name))
      )
    }
    schema
  }

  override def copy(extra: ParamMap): CardinalityLimitColumnPrunerTransformer =
    defaultCopy(extra)
}

object CardinalityLimitColumnPrunerTransformer
    extends DefaultParamsReadable[CardinalityLimitColumnPrunerTransformer] {
  override def load(path: String): CardinalityLimitColumnPrunerTransformer =
    super.load(path)
}
