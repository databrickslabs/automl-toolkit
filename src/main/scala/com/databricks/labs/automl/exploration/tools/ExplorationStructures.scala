package com.databricks.labs.automl.exploration.tools

case class OneDimStatsData(mean: Double,
                           geomMean: Double,
                           variance: Double,
                           semiVariance: Double,
                           stddev: Double,
                           skew: Double,
                           kurtosis: Double,
                           kurtosisType: String,
                           skewType: String)

case class ShapiroScoreData(w: Double, z: Double, probability: Double)

case class ShapiroInternalData(w: Double,
                               z: Double,
                               probability: Double,
                               normalcyTest: Boolean,
                               normalcy: String)

case class RegressionInternal(sumX: Double,
                              sumY: Double,
                              sumSqX: Double,
                              sumSqY: Double,
                              sumProduct: Double)

case class RegressionCoefficients(slope: Double,
                                  intercept: Double,
                                  t1: Double,
                                  t2: Double,
                                  t3: Double)

case class RegressionBarData(xBar: Double, yBar: Double, xyBar: Double)
case class RegressionResidualData(ssr: Double, rss: Double)

case class SimpleRegressorResult(slope: Double,
                                 slopeStdErr: Double,
                                 slopeConfidenceInterval: Double,
                                 intercept: Double,
                                 interceptStdErr: Double,
                                 rSquared: Double,
                                 significance: Double,
                                 mse: Double,
                                 rmse: Double,
                                 sumSquares: Double,
                                 totalSumSquares: Double,
                                 sumSquareError: Double,
                                 pairLength: Long,
                                 pearsonR: Double,
                                 crossProductSum: Double)
