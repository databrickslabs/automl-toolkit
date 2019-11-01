package com.databricks.labs.automl.model.tools.structures

import com.databricks.labs.automl.params.MLPCConfig

case class NumericBoundaries(minimum: Double, maximum: Double)

case class NumericArrayCollection(selectedPayload: Array[Double],
                                  remainingPayload: Array[Array[Double]])
case class StringSelectionReturn(selectedStringValue: String,
                                 IndexCounterStatus: Int)

case class PermutationConfiguration(
  modelType: String,
  permutationTarget: Int,
  numericBoundaries: Map[String, (Double, Double)],
  stringBoundaries: Map[String, List[String]]
)

case class MLPCPermutationConfiguration(
  permutationTarget: Int,
  numericBoundaries: Map[String, (Double, Double)],
  stringBoundaries: Map[String, List[String]],
  inputFeatureSize: Int,
  distinctClasses: Int
)

// RANDOM FOREST
case class RandomForestPermutationCollection(
  numTreesArray: Array[Double],
  maxBinsArray: Array[Double],
  maxDepthArray: Array[Double],
  minInfoGainArray: Array[Double],
  subSamplingRateArray: Array[Double],
  impurityArray: Array[String],
  featureSubsetStrategyArray: Array[String]
)

case class RandomForestNumericArrays(numTreesArray: Array[Double],
                                     maxBinsArray: Array[Double],
                                     maxDepthArray: Array[Double],
                                     minInfoGainArray: Array[Double],
                                     subSamplingRateArray: Array[Double])

case class RandomForestModelRunReport(numTrees: Int,
                                      impurity: String,
                                      maxBins: Int,
                                      maxDepth: Int,
                                      minInfoGain: Double,
                                      subSamplingRate: Double,
                                      featureSubsetStrategy: String,
                                      score: Double)

//DECISION TREES
case class TreesPermutationCollection(impurityArray: Array[String],
                                      maxBinsArray: Array[Double],
                                      maxDepthArray: Array[Double],
                                      minInfoGainArray: Array[Double],
                                      minInstancesPerNodeArray: Array[Double])

case class TreesNumericArrays(maxBinsArray: Array[Double],
                              maxDepthArray: Array[Double],
                              minInfoGainArray: Array[Double],
                              minInstancesPerNodeArray: Array[Double])

case class TreesModelRunReport(impurity: String,
                               maxBins: Int,
                               maxDepth: Int,
                               minInfoGain: Double,
                               minInstancesPerNode: Double,
                               score: Double)

//GRADIENT BOOSTED TREES
case class GBTPermutationCollection(impurityArray: Array[String],
                                    lossTypeArray: Array[String],
                                    maxBinsArray: Array[Double],
                                    maxDepthArray: Array[Double],
                                    maxIterArray: Array[Double],
                                    minInfoGainArray: Array[Double],
                                    minInstancesPerNodeArray: Array[Double],
                                    stepSizeArray: Array[Double])

case class GBTNumericArrays(maxBinsArray: Array[Double],
                            maxDepthArray: Array[Double],
                            maxIterArray: Array[Double],
                            minInfoGainArray: Array[Double],
                            minInstancesPerNodeArray: Array[Double],
                            stepSizeArray: Array[Double])

case class GBTModelRunReport(impurity: String,
                             lossType: String,
                             maxBins: Int,
                             maxDepth: Int,
                             maxIter: Int,
                             minInfoGain: Double,
                             minInstancesPerNode: Double,
                             stepSize: Double,
                             score: Double)

//LINEAR REGRESSION
case class LinearRegressionPermutationCollection(
  elasticNetParamsArray: Array[Double],
  fitInterceptArray: Array[Boolean],
  lossArray: Array[String],
  maxIterArray: Array[Double],
  regParamArray: Array[Double],
  standardizationArray: Array[Boolean],
  toleranceArray: Array[Double]
)

case class LinearRegressionNumericArrays(elasticNetParamsArray: Array[Double],
                                         maxIterArray: Array[Double],
                                         regParamArray: Array[Double],
                                         toleranceArray: Array[Double])

case class LinearRegressionModelRunReport(elasticNetParams: Double,
                                          fitIntercept: Boolean,
                                          loss: String,
                                          maxIter: Int,
                                          regParam: Double,
                                          standardization: Boolean,
                                          tolerance: Double,
                                          score: Double)

//LOGISTIC REGRESSION
case class LogisticRegressionPermutationCollection(
  elasticNetParamsArray: Array[Double],
  fitInterceptArray: Array[Boolean],
  maxIterArray: Array[Double],
  regParamArray: Array[Double],
  standardizationArray: Array[Boolean],
  toleranceArray: Array[Double]
)

case class LogisticRegressionNumericArrays(elasticNetParamsArray: Array[Double],
                                           maxIterArray: Array[Double],
                                           regParamArray: Array[Double],
                                           toleranceArray: Array[Double])

case class LogisticRegressionModelRunReport(elasticNetParams: Double,
                                            fitIntercept: Boolean,
                                            maxIter: Int,
                                            regParam: Double,
                                            standardization: Boolean,
                                            tolerance: Double,
                                            score: Double)

//SVM
case class SVMPermutationCollection(fitInterceptArray: Array[Boolean],
                                    maxIterArray: Array[Double],
                                    regParamArray: Array[Double],
                                    standardizationArray: Array[Boolean],
                                    toleranceArray: Array[Double])

case class SVMNumericArrays(maxIterArray: Array[Double],
                            regParamArray: Array[Double],
                            toleranceArray: Array[Double])

case class SVMModelRunReport(fitIntercept: Boolean,
                             maxIter: Int,
                             regParam: Double,
                             standardization: Boolean,
                             tolerance: Double,
                             score: Double)

//MLPC

case class MLPCGenerator(layerCount: Int,
                         hiddenLayerSizeAdjust: Int,
                         maxIter: Int,
                         solver: String,
                         stepSize: Double,
                         tolerance: Double)

case class MLPCPermutationCollection(layerCountArray: Array[Int],
                                     layersArray: Array[Array[Int]],
                                     maxIterArray: Array[Double],
                                     solverArray: Array[String],
                                     stepSizeArray: Array[Double],
                                     toleranceArray: Array[Double],
                                     hiddenLayerSizeAdjustArray: Array[Int])

case class MLPCModelingConfig(layerCount: Int,
                              layers: Array[Int],
                              maxIter: Int,
                              solver: String,
                              stepSize: Double,
                              tolerance: Double,
                              hiddenLayerSizeAdjust: Int)

case class MLPCNumericArrays(layersArray: Array[Array[Int]],
                             maxIterArray: Array[Double],
                             stepSizeArray: Array[Double],
                             toleranceArray: Array[Double])

case class MLPCModelRunReport(layers: Int,
                              maxIter: Int,
                              solver: String,
                              stepSize: Double,
                              tolerance: Double,
                              hiddenLayerSizeAdjust: Int,
                              score: Double)

case class MLPCArrayCollection(selectedPayload: MLPCConfig,
                               remainingPayloads: MLPCNumericArrays)

//XGBOOST
case class XGBoostPermutationCollection(alphaArray: Array[Double],
                                        etaArray: Array[Double],
                                        gammaArray: Array[Double],
                                        lambdaArray: Array[Double],
                                        maxDepthArray: Array[Double],
                                        subSampleArray: Array[Double],
                                        minChildWeightArray: Array[Double],
                                        numRoundArray: Array[Double],
                                        maxBinsArray: Array[Double],
                                        trainTestRatioArray: Array[Double])

case class XGBoostNumericArrays(alphaArray: Array[Double],
                                etaArray: Array[Double],
                                gammaArray: Array[Double],
                                lambdaArray: Array[Double],
                                maxDepthArray: Array[Double],
                                subSampleArray: Array[Double],
                                minChildWeightArray: Array[Double],
                                numRoundArray: Array[Double],
                                maxBinsArray: Array[Double],
                                trainTestRatioArray: Array[Double])

case class XGBoostModelRunReport(alpha: Double,
                                 eta: Double,
                                 gamma: Double,
                                 lambda: Double,
                                 maxDepth: Int,
                                 subSample: Double,
                                 minChildWeight: Double,
                                 numRound: Int,
                                 maxBins: Int,
                                 trainTestRatio: Double,
                                 score: Double)

//LightGBM
case class LightGBMPermutationCollection(
  baggingFractionArray: Array[Double],
  baggingFreqArray: Array[Double],
  featureFractionArray: Array[Double],
  learningRateArray: Array[Double],
  maxBinArray: Array[Double],
  maxDepthArray: Array[Double],
  minSumHessianInLeafArray: Array[Double],
  numIterationsArray: Array[Double],
  numLeavesArray: Array[Double],
  boostFromAverageArray: Array[Boolean],
  lambdaL1Array: Array[Double],
  lambdaL2Array: Array[Double],
  alphaArray: Array[Double],
  boostingTypeArray: Array[String]
)

case class LightGBMNumericArrays(baggingFractionArray: Array[Double],
                                 baggingFreqArray: Array[Double],
                                 featureFractionArray: Array[Double],
                                 learningRateArray: Array[Double],
                                 maxBinArray: Array[Double],
                                 maxDepthArray: Array[Double],
                                 minSumHessianInLeafArray: Array[Double],
                                 numIterationsArray: Array[Double],
                                 numLeavesArray: Array[Double],
                                 lambdaL1Array: Array[Double],
                                 lambdaL2Array: Array[Double],
                                 alphaArray: Array[Double])

case class LightGBMModelRunReport(baggingFraction: Double,
                                  baggingFreq: Int,
                                  featureFraction: Double,
                                  learningRate: Double,
                                  maxBin: Int,
                                  maxDepth: Int,
                                  minSumHessianInLeaf: Double,
                                  numIterations: Int,
                                  numLeaves: Int,
                                  boostFromAverage: Boolean,
                                  lambdaL1: Double,
                                  lambdaL2: Double,
                                  alpha: Double,
                                  boostingType: String,
                                  score: Double)
