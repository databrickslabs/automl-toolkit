package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.ensemble.tuner.impl._
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, MainConfig}

private[tuner] case class ModelFamilyTunerTypes(modelFamily: Array[String], tunerType: Class[_ <: TunerDelegator])

private[tuner] object ModelFamilyTunerType extends Enumeration {
  type ModelFamilyTunerType = ModelFamilyTunerTypes
  val RANDOM_FOREST: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("RandomForest"), classOf[RandomForestTunerDelegator])
  val XG_BOOST: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("XGBoost"), classOf[XGBoostTunerDelegator])
  val GBM: ModelFamilyTunerTypes = ModelFamilyTunerTypes(
    Array(
    "gbmBinary", "gbmMulti", "gbmMultiOVA", "gbmHuber", "gbmFair",
    "gbmLasso", "gbmRidge", "gbmPoisson", "gbmQuantile", "gbmMape",
    "gbmTweedie", "gbmGamma"),
    classOf[GbmTunerDelegator]
  )
  val GBT: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("GBT"), classOf[GbtTunerDelegator])
  val LINEAR_REGRESSION: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("LinearRegression"), classOf[LinearRegressionTunerDelegator])
  val LOGISTIC_REGRESSION: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("LogisticRegression"), classOf[LogisticRegressionTunerDelegator])
  val SVM: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("SVM"), classOf[SvmTunerDelegator])
  val TREES: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("Trees"), classOf[TreesTunerDelegator])

  def getTunerInstanceByModelFamily(modelFamily: String,
                                    mainConfig: MainConfig,
                                    payload: DataGeneration,
                                    testTrainSplitData: Array[TrainSplitReferences]): TunerDelegator = {
    ModelFamilyTunerType
      .values
      .map(
      _.asInstanceOf[ModelFamilyTunerTypes])
      .filter(_.modelFamily.contains(modelFamily))
      .head
      .tunerType
      .getConstructor(classOf[MainConfig], classOf[DataGeneration], classOf[Array[TrainSplitReferences]])
      .newInstance(mainConfig, payload, testTrainSplitData)
  }
}
