package com.databricks.spark.automatedml.model.tools.structures

case class NumericBoundaries(
                              minimum: Double,
                              maximum: Double
                            )

case class NumericArrayCollection(
                                   selectedPayload: Array[Double],
                                   remainingPayload: Array[Array[Double]]
                                 )
case class StringSelectionReturn(
                                  selectedStringValue: String,
                                  IndexCounterStatus: Int
                                )

case class PermutationConfiguration(
                                                permutationTarget: Int,
                                                numericBoundaries: Map[String, (Double, Double)],
                                                stringBoundaries: Map[String, List[String]]
                                               )

case class RandomForestPermutationCollection(
                                              numTreesArray: Array[Double],
                                              maxBinsArray: Array[Double],
                                              maxDepthArray: Array[Double],
                                              minInfoGainArray: Array[Double],
                                              subSamplingRateArray: Array[Double],
                                              impurityArray: Array[String],
                                              featureSubsetStrategyArray: Array[String]
                                            )


case class RandomForestNumericArrays(
                                      numTreesArray: Array[Double],
                                      maxBinsArray: Array[Double],
                                      maxDepthArray: Array[Double],
                                      minInfoGainArray: Array[Double],
                                      subSamplingRateArray: Array[Double]
                                    )

case class RandomForestModelRunReport(
                                       numTrees: Int,
                                       impurity: String,
                                       maxBins: Int,
                                       maxDepth: Int,
                                       minInfoGain: Double,
                                       subSamplingRate: Double,
                                       featureSubsetStrategy: String,
                                       score: Double
                                     )