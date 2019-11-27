package org.apache.spark.ml.automl.feature

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{ParamMap, Params, StringArrayParam}
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCols}
import org.apache.spark.ml.util.{
  DefaultParamsWritable,
  Identifiable,
  MLReadable,
  MLReader,
  MLWritable,
  MLWriter
}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

trait FeatureInteractorBase
    extends Params
    with HasInputCols
    with HasOutputCols {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    ???
  }

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
class FeatureInteractorModel(override val uid: String)
    extends Model[FeatureInteractorModel]
    with FeatureInteractorBase
    with MLWritable {

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
