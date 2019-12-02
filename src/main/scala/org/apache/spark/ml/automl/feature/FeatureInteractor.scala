package org.apache.spark.ml.automl.feature

import org.apache.spark.ml.param.shared.HasOutputCols
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

trait FeatureInteractorBase
    extends Params
    with HasNominalColumns
    with HasContinuousColumns {

  final val modelingType: Param[String] = new Param[String](
    this,
    "modelingType",
    "Modeling type: either 'regressor' or 'classifier'",
    ParamValidators.inArray(FeatureInteractor.supportedModelTypes)
  )

  def setModelingType(value: String): this.type = set(modelingType, value)
  def getModelingType: String = $(modelingType)

  final val retentionMode: Param[String] = new Param[String](
    this,
    "retentionMode",
    "One of: 'all', 'optimistic', or 'strict' for interacted field inclusion", ParamValidators.inArray(FeatureInteractor.supportedRetentionModes)
  )

  def setRetentionMode(value: String): this.type = set(retentionMode, value)
  def getRetentionMode: String = $(retentionMode)

  final val continuousDiscretizerBucketCount: IntParam = new IntParam(this, "continuousDiscretizerBucketCount", )

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    ???
  }

}

trait HasContinuousColumns extends Params {

  final val continuousColumns: StringArrayParam = new StringArrayParam(
    this,
    "continuousColumns",
    "Continuous Columns to be included in feature vector"
  )

  def setContinuousColumns(value: Array[String]): this.type =
    set(continuousColumns, value)

  def getContinuousColumns: Array[String] = $(continuousColumns)

}

trait HasNominalColumns extends Params {

  final val nominalColumns: StringArrayParam = new StringArrayParam(
    this,
    "nominalColumns",
    "Nominal Columns to be included in feature vector"
  )

  def setNominalColumns(value: Array[String]): this.type =
    set(nominalColumns, value)

  def getNominalColumns: Array[String] = $(nominalColumns)

}

class FeatureInteractor(override val uid: String)
    extends Estimator[FeatureInteractorModel]
    with DefaultParamsWritable
    with HasNominalColumns
    with HasOutputCols
    with FeatureInteractorBase {

  def this() = this(Identifiable.randomUID("featureInteractor"))

  override def copy(extra: ParamMap): FeatureInteractor = defaultCopy(extra)

  override def fit(dataset: Dataset[_]): FeatureInteractorModel = ???

  override def transformSchema(schema: StructType): StructType = ???

}

object FeatureInteractor extends DefaultParamsReadable[FeatureInteractor] {

  private[feature] val CLASSIFIER: String = "classifier"
  private[feature] val REGRESSOR: String = "regressor"
  private[feature] val supportedModelTypes: Array[String] =
    Array(CLASSIFIER, REGRESSOR)

  private[feature] val ALL: String = "all"
  private[feature] val OPTIMISTIC: String = "optimistic"
  private[feature] val STRICT: String = "strict"
  private[feature] val supportedRetentionModes: Array[String] =
    Array(ALL, OPTIMISTIC, STRICT)

}

class FeatureInteractorModel(override val uid: String)
    extends Model[FeatureInteractorModel]
    with FeatureInteractorBase
    with MLWritable {
  // Setters and Getters

  override def copy(extra: ParamMap): FeatureInteractorModel = ???

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def transformSchema(schema: StructType): StructType = ???

  override def write: MLWriter = ???

}
object FeatureInteractorModel extends MLReadable[FeatureInteractorModel] {

  private[FeatureInteractorModel] class FeatureInteractorModelWriter(
    instance: FeatureInteractorModel
  ) extends MLWriter {

    override protected def saveImpl(path: String) = ???

  }

  private class FeatureInteractorModelReader
      extends MLReader[FeatureInteractorModel] {
    override def load(path: String): FeatureInteractorModel = ???
  }

  override def read: MLReader[FeatureInteractorModel] = ???

  override def load(path: String): FeatureInteractorModel = super.load(path)
}
