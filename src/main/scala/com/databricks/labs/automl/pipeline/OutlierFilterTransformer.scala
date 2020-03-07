package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.sanitize.OutlierFiltering
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * @author Jas Bali
  * This transformer wraps [[OutlierFiltering]] in a transform method
  * @param uid
  */
class OutlierFilterTransformer(override val uid: String)
  extends AbstractTransformer
    with DefaultParamsWritable
    with HasLabelColumn
    with HasFieldsToIgnore
    with IsTrainingStage {

  @transient lazy private val logger: Logger = Logger.getLogger(this.getClass)

  def this() = {
    this(Identifiable.randomUID("OutlierFilterTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setFieldsToIgnore(Array.empty)
    setDebugEnabled(false)
  }

  final val filterBounds: Param[String] = new Param[String](this, "filterBounds", "Filter Bounds")

  final val lowerFilterNTile: DoubleParam = new DoubleParam(this, "lowerFilterNTile", "lowerFilterNTile")

  final val upperFilterNTile: DoubleParam = new DoubleParam(this, "upperFilterNTile", "upperFilterNTile")

  final val filterPrecision: DoubleParam = new DoubleParam(this, "filterPrecision", "filterPrecision")

  final val parallelism: IntParam = new IntParam(this, "parallelism", "parallelism")

  final val continuousDataThreshold: IntParam = new IntParam(this, "continuousDataThreshold", "continuousDataThreshold")

  def setFilterBounds(value: String): this.type = set(filterBounds, value)

  def getFilterBounds: String = $(filterBounds)

  def setLowerFilterNTile(value: Double): this.type = set(lowerFilterNTile, value)

  def getLowerFilterNTile: Double = $(lowerFilterNTile)

  def setUpperFilterNTile(value: Double): this.type = set(upperFilterNTile, value)

  def getUpperFilterNTile: Double = $(upperFilterNTile)

  def setFilterPrecision(value: Double): this.type = set(filterPrecision, value)

  def getFilterPrecision: Double = $(filterPrecision)

  def setParallelism(value: Int): this.type = set(parallelism, value)

  def getParallelism: Int = $(parallelism)

  def setContinuousDataThreshold(value: Int): this.type = set(continuousDataThreshold, value)

  def getContinuousDataThreshold: Int = $(continuousDataThreshold)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    if(dataset.columns.contains(getLabelColumn)) {
      // Output has no feature vector
      val outlierFiltering = new OutlierFiltering(dataset.toDF())
        .setLabelCol(getLabelColumn)
        .setFilterBounds(getFilterBounds)
        .setLowerFilterNTile(getLowerFilterNTile)
        .setUpperFilterNTile(getUpperFilterNTile)
        .setFilterPrecision(getFilterPrecision)
        .setParallelism(getParallelism)
        .setContinuousDataThreshold(getContinuousDataThreshold)

      val (outlierCleanedData, outlierRemovedData, filteringMap) =
        outlierFiltering.filterContinuousOutliers(Array(getAutomlInternalId) ++ getFieldsToIgnore, getFieldsToIgnore)
      val outlierRemovalInfo =
        s"Removed outlier data.  Total rows removed = ${outlierRemovedData.count()}"
      logger.log(Level.INFO, outlierRemovalInfo)
      println(outlierRemovalInfo)
      //setInferenceOutlierMap(filteringMap)
      // Validate mutated Dfs
      validateMutatedf(dataset.toDF(), outlierCleanedData, outlierRemovedData)
      outlierCleanedData
    } else {
      dataset.toDF()
    }
  }

  private def validateMutatedf(originalDf: DataFrame,
                               mutatedDf: DataFrame,
                               outlierDf: DataFrame): Unit = {
    assert(
      originalDf.count() == mutatedDf.count() + outlierDf.count(),
      "Original Df count doesn't match the sum of outlier filter output data and removed data")
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): OutlierFilterTransformer = defaultCopy(extra)
}

object OutlierFilterTransformer extends DefaultParamsReadable[OutlierFilterTransformer] {
  override def load(path: String): OutlierFilterTransformer = super.load(path)
}
