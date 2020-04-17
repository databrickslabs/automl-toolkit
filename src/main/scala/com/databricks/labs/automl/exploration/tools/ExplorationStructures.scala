package com.databricks.labs.automl.exploration.tools

import org.apache.commons.math3.analysis.polynomials.PolynomialFunction
import org.apache.commons.math3.distribution.RealDistribution
import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.apache.spark.sql.DataFrame

case class SummaryStats(count: Long,
                        min: Double,
                        max: Double,
                        sum: Double,
                        mean: Double,
                        geometricMean: Double,
                        variance: Double,
                        popVariance: Double,
                        secondMoment: Double,
                        sumOfSquares: Double,
                        stdDeviation: Double,
                        sumOfLogs: Double)

case class DistributionValidationData(test: String,
                                      pValue: Double,
                                      dStatistic: Double)

case class DistributionTestPayload(testName: String,
                                   distribution: RealDistribution)

case class DistributionTestResult(bestDistributionFit: String,
                                  bestDistributionPValue: Double,
                                  bestDistributionDStatistic: Double,
                                  allTests: Array[DistributionValidationData])

case class OneDimStatsData(mean: Double,
                           geomMean: Double,
                           variance: Double,
                           semiVariance: Double,
                           stddev: Double,
                           skew: Double,
                           kurtosis: Double,
                           kurtosisType: String,
                           skewType: String,
                           summaryStats: SummaryStats,
                           distributionData: DistributionTestResult)

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

case class PolynomialRegressorResult(order: Int,
                                     function: PolynomialFunction,
                                     residualSumSquares: Double,
                                     sumSquareError: Double,
                                     totalSumSquares: Double,
                                     mse: Double,
                                     rmse: Double,
                                     r2: Double)

case class PairedSeq(left: SummaryStatistics, right: SummaryStatistics)
case class TTestData(alpha: Double,
                     tStat: Double,
                     tTestSignificance: Boolean,
                     tTestPValue: Double,
                     equivalencyJudgement: Char)

case class KSTestResult(ksTestPvalue: Double,
                        ksTestDStatistic: Double,
                        ksTestEquivalency: Char)

case class CorrelationTestResult(covariance: Double,
                                 pearsonCoefficient: Double,
                                 spearmanCoefficient: Double,
                                 kendallsTauCoefficient: Double)

case class PairedTestResult(correlationTestData: CorrelationTestResult,
                            tTestData: TTestData,
                            kolmogorovSmirnovData: KSTestResult)

case class PCAReducerResult(data: DataFrame,
                            explainedVariances: Array[Double],
                            pcMatrix: Array[PCACEigenResult],
                            pcEigenDataFrame: DataFrame)

case class PCACEigenResult(column: String,
                           PCA1EigenVector: Double,
                           PCA2EigenVector: Double)
