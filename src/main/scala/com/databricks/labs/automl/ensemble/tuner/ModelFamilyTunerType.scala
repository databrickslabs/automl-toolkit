package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.ensemble.tuner.impl._
import com.databricks.labs.automl.model.tools.ModelTypes.Trees
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, MainConfig}

private[tuner] case class ModelFamilyTunerTypes(modelFamily: Array[String], tunerType: Class[_ <: TunerDelegator]) extends Enumeration

private[tuner] object ModelFamilyTunerType {

  val RANDOM_FOREST: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("RandomForest"), classOf[RandomForestTunerDelegator])

  val XG_BOOST: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("XGBoost"), classOf[XGBoostTunerDelegator])

  /**
    * Removing reference to GBM model until we finish testing their Spark 3.x distribution
  val GBM: ModelFamilyTunerTypes = ModelFamilyTunerTypes(
    Array(
    "gbmBinary", "gbmMulti", "gbmMultiOVA", "gbmHuber", "gbmFair",
    "gbmLasso", "gbmRidge", "gbmPoisson", "gbmQuantile", "gbmMape",
    "gbmTweedie", "gbmGamma"),
    classOf[GbmTunerDelegator]
  )
  */

  val GBT: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("GBT"), classOf[GbtTunerDelegator])

  val LINEAR_REGRESSION: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("LinearRegression"), classOf[LinearRegressionTunerDelegator])

  val LOGISTIC_REGRESSION: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("LogisticRegression"), classOf[LogisticRegressionTunerDelegator])

  val SVM: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("SVM"), classOf[SvmTunerDelegator])

  val TREES: ModelFamilyTunerTypes = ModelFamilyTunerTypes(Array("Trees"), classOf[TreesTunerDelegator])

  def getTunerInstanceByModelFamily(modelFamily: String,
                                    mainConfig: MainConfig,
                                    payload: DataGeneration,
                                    testTrainSplitData: Array[TrainSplitReferences]): TunerDelegator = {
    val tunerType = modelFamily match {
      case "RandomForest" => RANDOM_FOREST.tunerType
      case "XGBoost" => XG_BOOST.tunerType
      //GBM specific
      //Remove GBM reference until completion of Spark 3.x testing
//      case "gbmBinary" => GBM.tunerType
//      case "gbmMulti" => GBM.tunerType
//      case "gbmMultiOVA" => GBM.tunerType
//      case "gbmHuber" => GBM.tunerType
//      case "gbmFair" => GBM.tunerType
//      case "gbmLasso" => GBM.tunerType
//      case "gbmRidge" => GBM.tunerType
//      case "gbmPoisson" => GBM.tunerType
//      case "gbmQuantile" => GBM.tunerType
//      case "gbmMape" => GBM.tunerType
//      case "gbmTweedie" => GBM.tunerType
//      case "gbmGamma" => GBM.tunerType

      case "GBT" => GBT.tunerType
      case "LinearRegression" => LINEAR_REGRESSION.tunerType
      case "LogisticRegression" => LOGISTIC_REGRESSION.tunerType
      case "SVM" => SVM.tunerType
      case "Trees" => TREES.tunerType
    }

    tunerType
      .getConstructor(classOf[MainConfig], classOf[DataGeneration], classOf[Array[TrainSplitReferences]])
      .newInstance(mainConfig, payload, testTrainSplitData)
  }
}
