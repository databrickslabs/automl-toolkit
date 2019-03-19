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



case class RandomForestPermutationCollection(
                                              numTreesArray: Array[Double],
                                              maxBinsArray: Array[Double],
                                              maxDepthArray: Array[Double],
                                              minInfoGainArray: Array[Double],
                                              subSamplingRateArray: Array[Double],
                                              impurityArray: Array[String],
                                              featureSubsetStrategyArray: Array[String]
                                            )

case class RandomForestPermutationConfiguration(
                                                permutationTarget: Int,
                                                numericBoundaries: Map[String, (Double, Double)],
                                                stringBoundaries: Map[String, List[String]]
                                               )

case class RandomForestNumericArrays(
                                      numTreesArray: Array[Double],
                                      maxBinsArray: Array[Double],
                                      maxDepthArray: Array[Double],
                                      minInfoGainArray: Array[Double],
                                      subSamplingRateArray: Array[Double]
                                    )