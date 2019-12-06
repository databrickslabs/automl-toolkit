package com.databricks.labs.automl.pipeline
import com.databricks.labs.automl.sanitize.FeatureCorrelationDetection
import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, SchemaUtils}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.param.{DoubleParam, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
/**
  * @author Jas Bali
  * A transformer stage that wraps [[FeatureCorrelationDetection]] in the transform method.
  */
class CovarianceFilterTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFeaturesColumns
    with HasFieldsRemoved
    with HasTransformCalculated
    with HasFeatureColumn {
  private val logger: Logger = Logger.getLogger(this.getClass)
  def this() = {
    this(Identifiable.randomUID("CovarianceFilterTransformer"))
    setFieldsRemoved(Array.empty)
    setTransformCalculated(false)
    setCorrelationCutoffLow(-0.99)
    setCorrelationCutoffHigh(0.99)
    setFeatureColumns(Array.empty)
    setDebugEnabled(false)
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
  }

  final val correlationCutoffLow: DoubleParam = new DoubleParam(this, "correlationCutoffLow", "correlationCutoffLow")
  final val correlationCutoffHigh: DoubleParam = new DoubleParam(this, "correlationCutoffHigh", "correlationCutoffHigh")
  def setCorrelationCutoffLow(value: Double): this.type = set(correlationCutoffLow, value)
  def getCorrelationCutoffLow: Double = $(correlationCutoffLow)
  def setCorrelationCutoffHigh(value: Double): this.type = set(correlationCutoffHigh, value)
  def getCorrelationCutoffHigh: Double = $(correlationCutoffHigh)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    if(dataset.columns.contains(getLabelColumn)) {
      if (SchemaUtils.isNotEmpty(getFeatureColumns)) {
        setFeatureColumns(dataset.columns.filterNot(item => Array(getLabelColumn, AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL).contains(item)))
      }
      // Output has no feature vector
      if (!getTransformCalculated) {
        val covarianceFilteredData =
          new FeatureCorrelationDetection(dataset.toDF(), getFeatureColumns.filterNot(item => AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL.equals(item) || getFeatureCol.equals(item)))
            .setLabelCol(getLabelColumn)
            .setCorrelationCutoffLow(getCorrelationCutoffLow)
            .setCorrelationCutoffHigh(getCorrelationCutoffHigh)
            .filterFeatureCorrelation()
        setFieldsRemoved(getFeatureColumns.filterNot(field => covarianceFilteredData.columns.contains(field)))
        setTransformCalculated(true)
        val covarianceFilterLog =
          s"Covariance Filtering completed.\n  Removed fields: ${getFieldsRemoved.mkString(", ")}"
        logger.log(Level.INFO, covarianceFilterLog)
        println(covarianceFilterLog)
        return covarianceFilteredData
      }
    }
    dataset.drop(getFieldsRemoved: _*)
  }
  override def transformSchemaInternal(schema: StructType): StructType = {
    if(schema.fieldNames.contains(getLabelColumn)) {
      StructType(schema.fields.filterNot(field => getFieldsRemoved.contains(field.name)))
    } else {
      schema
    }
  }
  override def copy(extra: ParamMap): CovarianceFilterTransformer = defaultCopy(extra)
}
object CovarianceFilterTransformer extends DefaultParamsReadable[CovarianceFilterTransformer] {
  override def load(path: String): CovarianceFilterTransformer = super.load(path)
}