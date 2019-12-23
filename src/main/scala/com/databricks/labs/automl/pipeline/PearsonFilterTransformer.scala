package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.sanitize.PearsonFiltering
import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, SchemaUtils}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap}
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsWritable,
  Identifiable
}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * @author Jas Bali
  * This transformer wraps [[PearsonFiltering]] in a transform method
  * @param uid
  */
class PearsonFilterTransformer(override val uid: String)
    extends AbstractTransformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeatureColumn
    with HasFeaturesColumns
    with HasFieldsRemoved
    with HasTransformCalculated {

  private val logger: Logger = Logger.getLogger(this.getClass)

  def this() = {
    this(Identifiable.randomUID("PearsonFilterTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setFieldsRemoved(Array.empty)
    setTransformCalculated(false)
    setDebugEnabled(false)
  }

  final val modelType: Param[String] =
    new Param[String](this, "modelType", "modelType")

  final val filterStatistic: Param[String] =
    new Param[String](this, "filterStatistic", "filterStatistic")

  final val filterDirection: Param[String] =
    new Param[String](this, "filterDirection", "filterDirection")

  final val filterManualValue: DoubleParam =
    new DoubleParam(this, "filterManualValue", "filterManualValue")

  final val filterMode: Param[String] =
    new Param[String](this, "filterMode", "filterMode")

  final val autoFilterNTile: DoubleParam =
    new DoubleParam(this, "autoFilterNTile", "autoFilterNTile")

  def setModelType(value: String): this.type = set(modelType, value)

  def setFilterStatistic(value: String): this.type = set(filterStatistic, value)

  def getFilterStatistic: String = $(filterStatistic)

  def setFilterDirection(value: String): this.type = set(filterDirection, value)

  def getFilterDirection: String = $(filterDirection)

  def setFilterManualValue(value: Double): this.type =
    set(filterManualValue, value)

  def getModelType: String = $(modelType)

  def getFilterManualValue: Double = $(filterManualValue)

  def setFilterMode(value: String): this.type = set(filterMode, value)

  def getFilterMode: String = $(filterMode)

  def setAutoFilterNTile(value: Double): this.type = set(autoFilterNTile, value)

  def getAutoFilterNTile: Double = $(autoFilterNTile)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    if (dataset.columns.contains(getLabelColumn)) {
      if (SchemaUtils.isNotEmpty(getFeatureColumns)) {
        setFeatureColumns(
          dataset.columns.filterNot(
            item => Array(getLabelColumn, getAutomlInternalId).contains(item)
          )
        )
      }
      if (!getTransformCalculated) {
        // Requires a DataFrame that has a feature vector field.  Output has no feature vector.
        val pearsonFiltering =
          new PearsonFiltering(dataset.toDF(), getFeatureColumns, getModelType)
            .setLabelCol(getLabelColumn)
            .setFeaturesCol(getFeatureCol)
            .setFilterStatistic(getFilterStatistic)
            .setFilterDirection(getFilterDirection)
            .setFilterManualValue(getFilterManualValue)
            .setFilterMode(getFilterMode)
            .setAutoFilterNTile(getAutoFilterNTile)
            .filterFields(Array(getAutomlInternalId))

        val removedFields = getFeatureColumns
          .filterNot(
            field => pearsonFiltering.schema.fieldNames.contains(field)
          )

        val pearsonFilterLog =
          s"Pearson Filtering completed.\n Removed fields: ${removedFields.mkString(", ")}"
        logger.log(Level.INFO, pearsonFiltering)
        println(pearsonFilterLog)

        setFieldsRemoved(removedFields)
        setTransformCalculated(true)
        return pearsonFiltering
      }
    }
    dataset.drop(getFieldsRemoved: _*)
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    if (schema.fieldNames.contains(getLabelColumn)) {
      StructType(
        schema.fields.filterNot(field => getFieldsRemoved.contains(field.name))
      )
    } else {
      schema
    }
  }

  override def copy(extra: ParamMap): PearsonFilterTransformer =
    defaultCopy(extra)
}

object PearsonFilterTransformer
    extends DefaultParamsReadable[PearsonFilterTransformer] {
  override def load(path: String): PearsonFilterTransformer = super.load(path)
}
