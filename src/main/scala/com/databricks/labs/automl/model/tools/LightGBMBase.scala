package com.databricks.labs.automl.model.tools

import com.databricks.labs.automl.exceptions.LightGBMModelTypeException

import scala.language.implicitConversions

object GBMTypes extends Enumeration {
  val GBMHuber = GBM("gbmHuber", "regressor")
  val GBMFair = GBM("gbmFair", "regressor")
  val GBMLasso = GBM("gbmLasso", "regressor")
  val GBMRidge = GBM("gbmRidge", "regressor")
  val GBMPoisson = GBM("gbmPoisson", "regressor")
  val GBMQuantile = GBM("gbmQuantile", "regressor")
  val GBMMape = GBM("gbmMape", "regressor")
  val GBMTweedie = GBM("gbmTweedie", "regressor")
  val GBMGamma = GBM("gbmGamma", "regressor")
  val GBMBinary = GBM("gbmBinary", "classifier")
  val GBMMulti = GBM("gbmMulti", "classifier")
  val GBMMultiOVA = GBM("gbmMultiOVA", "classifier")
  protected case class GBM(gbmType: String, modelType: String)
      extends super.Val()
  implicit def convert(value: Value): GBM = value.asInstanceOf[GBM]
}

object InitialGenerationMode extends Enumeration {
  type InitialGenerationMode = Value
  val RANDOM, PERMUTATIONS = Value

}

trait LightGBMBase {

  import GBMTypes._
  import InitialGenerationMode._

  final val allowableLightGBMRegressorTypes = Array(
    "gbmHuber",
    "gbmFair",
    "gbmLasso",
    "gbmRidge",
    "gbmPoisson",
    "gbmQuantile",
    "gbmMape",
    "gbmTweedie",
    "gbmGamma"
  )
  final val allowableLightGBMClassifierTypes =
    Array("gbmBinary", "gbmMulti", "gbmMultiOVA")

  final val BARRIER_MODE = false
  final val TIMEOUT = 36000

  protected[model] def getGBMType(modelSelection: String,
                                  lightGBMType: String): GBMTypes.Value = {

    (modelSelection, lightGBMType) match {
      case ("classifier", "gbmBinary")   => GBMBinary
      case ("classifier", "gbmMulti")    => GBMMulti
      case ("classifier", "gbmMultiOVA") => GBMMultiOVA
      case ("regressor", "gbmHuber")     => GBMHuber
      case ("regressor", "gbmFair")      => GBMFair
      case ("regressor", "gbmLasso")     => GBMLasso
      case ("regressor", "gbmRidge")     => GBMRidge
      case ("regressor", "gbmPoisson")   => GBMPoisson
      case ("regressor", "gbmQuantile")  => GBMQuantile
      case ("regressor", "gbmMape")      => GBMMape
      case ("regressor", "gbmTweedie")   => GBMTweedie
      case ("regressor", "gbmGamma")     => GBMGamma
      case _ =>
        throw LightGBMModelTypeException(
          modelSelection,
          lightGBMType,
          allowableLightGBMRegressorTypes,
          allowableLightGBMClassifierTypes
        )
    }
  }

  protected[model] def getInitialGenMode(
    mode: String
  ): InitialGenerationMode = {
    mode match {
      case "random"       => RANDOM
      case "permutations" => PERMUTATIONS
    }
  }

}

/**

//    https://sites.google.com/view/lauraepp/parameters

Regressor ->

alpha -> Double huber loss and quantile regression default: 0.9



Classifier ->

baggingFraction -> Double 0:1 (random bagging selection) default 1.0
baggingFreq -> Int (perform baggging at every k interval) default: 0:10?
baggingSeed -> Int -> Default 3
featureFraction -> Double 0:1 can be used to speed up training and prevent overfitting
lambdaL1 -> Double >=0.0 sets l1 regularization (lasso) default 0.0
lambdaL2 -> Double >=0.0 sets l2 regularization (ridge) default 0.0
learningRate -> Double 0:1 default 0.1
maxBin -> Int compression efficiency and lower values can prevent overfitting. default 255
maxDepth -> Int control the maximum depth of trees default: -1  3:15?
minSumHessianInLeaf -> Double used to deal with overfitting LOG SCALE default 1e-3
numIterations -> Int built by class count * numIterations default 100
numLeaves -> Int maximum number of leaves in one tree default 31



boostFromAverage -> Boolean Adjusts Initial Score for faster convergence default: True


boostingType: String -> gbdt, rf, dart, goss default: gbdt
objective -> String
  Regression -> regression_l2, regression_l1, huber, fair, poisson, quantile, mape, gamma, tweedie
  Classification -> binary, multiclass, multiclassova




categoricalSlotNames ? List of Categorical Columns in the feature Vector (needed?)
earlyStoppingRound ? Set Early stopping for metric evaluation Int

isUnbalance -> Boolean, set if a Binary Classification problem is heavily skewed default: false
timeout -> default 1200 (might want to increase this)
useBarrierExecutionMode -> default False, might want to try True to speed things up?

  */
