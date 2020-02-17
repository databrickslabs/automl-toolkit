package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.feature.SyntheticFeatureGenerator
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{BooleanType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}


/**
  * @author Jas Bali
  * This transformer wraps [[SyntheticFeatureGenerator]]
  */
class SyntheticFeatureGenTransformer(override val uid: String)
  extends AbstractTransformer
    with HasLabelColumn
    with HasFeatureColumn
    with HasFieldsToIgnore
    with DefaultParamsWritable
    with IsTrainingStage {

  def this() = {
    this(Identifiable.randomUID("SyntheticFeatureGenTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setFieldsToIgnore(Array(getAutomlInternalId))
    setDebugEnabled(false)
  }

  final val syntheticCol: Param[String] = new Param[String](this, "syntheticCol", "syntheticCol")
  final val kGroups: IntParam = new IntParam(this, "kGroups", "kGroups")
  final val kMeansMaxIter: IntParam = new IntParam(this, "kMeansMaxIter", "kMeansMaxIter")
  final val kMeansTolerance: DoubleParam = new DoubleParam(this, "kMeansTolerance", "kMeansTolerance")
  final val kMeansDistanceMeasurement: Param[String] = new Param[String](this, "kMeansDistanceMeasurement", "kMeansDistanceMeasurement")
  final val kMeansSeed: LongParam = new LongParam(this, "kMeansSeed", "kMeansSeed")
  final val kMeansPredictionCol: Param[String] = new Param[String](this, "kMeansPredictionCol", "kMeansPredictionCol")
  final val lshHashTables: IntParam = new IntParam(this, "lshHashTables", "lshHashTables")
  final val lshSeed: LongParam = new LongParam(this, "lshSeed", "lshSeed")
  final val lshOutputCol: Param[String] = new Param[String](this, "lshOutputCol", "lshOutputCol")
  final val quorumCount: IntParam = new IntParam(this, "quorumCount", "quorumCount")
  final val minimumVectorCountToMutate: IntParam = new IntParam(this, "minimumVectorCountToMutate", "minimumVectorCountToMutate")
  final val vectorMutationMethod: Param[String] = new Param[String](this, "vectorMutationMethod", "vectorMutationMethod")
  final val mutationMode: Param[String] = new Param[String](this, "mutationMode", "mutationMode")
  final val mutationValue: DoubleParam = new DoubleParam(this, "mutationValue", "mutationValue")
  final val labelBalanceMode: Param[String] = new Param[String](this, "labelBalanceMode", "labelBalanceMode")
  final val cardinalityThreshold: IntParam = new IntParam(this, "cardinalityThreshold", "cardinalityThreshold")
  final val numericRatio: DoubleParam = new DoubleParam(this, "numericRatio", "numericRatio")
  final val numericTarget: IntParam = new IntParam(this, "numericTarget", "numericTarget")

  def setSyntheticCol(value: String): this.type = set(syntheticCol, value)
  def setKGroups(value: Int): this.type = set(kGroups, value)
  def setKMeansMaxIter(value: Int): this.type = set(kMeansMaxIter, value)
  def setKMeansTolerance(value: Double): this.type = set(kMeansTolerance, value)
  def setKMeansDistanceMeasurement(value: String): this.type = set(kMeansDistanceMeasurement, value)
  def setKMeansSeed(value: Long): this.type = set(kMeansSeed, value)
  def setKMeansPredictionCol(value: String): this.type = set(kMeansPredictionCol, value)
  def setLshHashTables(value: Int): this.type = set(lshHashTables, value)
  def setLshSeed(value: Long): this.type = set(lshSeed, value)
  def setLshOutputCol(value: String): this.type = set(lshOutputCol, value)
  def setQuorumCount(value: Int): this.type = set(quorumCount, value)
  def setMinimumVectorCountToMutate(value: Int): this.type = set(minimumVectorCountToMutate, value)
  def setVectorMutationMethod(value: String): this.type = set(vectorMutationMethod, value)
  def setMutationMode(value: String): this.type = set(mutationMode, value)
  def setMutationValue(value: Double): this.type = set(mutationValue, value)
  def setLabelBalanceMode(value: String): this.type = set(labelBalanceMode, value)
  def setCardinalityThreshold(value: Int): this.type = set(cardinalityThreshold, value)
  def setNumericRatio(value: Double): this.type = set(numericRatio, value)
  def setNumericTarget(value: Int): this.type = set(numericTarget, value)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    transformSchemaInternal(dataset.schema)
    SyntheticFeatureGenerator(
        dataset.toDF(),
        getFeatureCol,
        getLabelColumn,
        $(syntheticCol),
        getFieldsToIgnore,
        $(kGroups),
        $(kMeansMaxIter),
        $(kMeansTolerance),
        $(kMeansDistanceMeasurement),
        $(kMeansSeed),
        $(kMeansPredictionCol),
        $(lshHashTables),
        $(lshSeed),
        $(lshOutputCol),
        $(quorumCount),
        $(minimumVectorCountToMutate),
        $(vectorMutationMethod),
        $(mutationMode),
        $(mutationValue),
        $(labelBalanceMode),
        $(cardinalityThreshold),
        $(numericRatio),
        $(numericTarget)
      )
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    StructType(schema.fields ++ Array(StructField($(syntheticCol), BooleanType, nullable = true)))
  }

  override def copy(extra: ParamMap): SyntheticFeatureGenTransformer = defaultCopy(extra)
}

object SyntheticFeatureGenTransformer extends DefaultParamsReadable[SyntheticFeatureGenTransformer] {
  override def load(path: String): SyntheticFeatureGenTransformer = super.load(path)
}
