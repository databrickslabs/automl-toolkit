package com.databricks.labs.automl.exploration.tools

import org.apache.commons.math3.distribution
import org.apache.commons.math3.distribution._
import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.apache.commons.math3.stat.descriptive.moment._
import org.apache.commons.math3.stat.inference.TestUtils

/**
  * Package for testing a one-dimensional data set with standard data explanation metrics
  *
  * Attributes tested:
  * Mean
  * Geometric Mean
  * Variance
  * Semi Variance
  * Standard Deviation
  * Skew
  * Kurtosis
  *
  * The return type also provides String classifications based on thresholds for Skew and Kurtosis to classify the
  * distribution.
  *
  * @note
  *       Kurtosis types:
  *       Mesokurtic - kurtosis around zero
  *       Leptokurtic - positive excess kurtosis (long heavy tails)
  *       Platykurtic - negative excess kurtosis (short thin tails)
  *
  *
  *       Skewness types:
  *       Symmetrical -> normal
  *       Asymmetricral Positive skewness -> right tailed
  *       Asymmetrical Negative skewness -> left tailed
  * @param data Array[Double] to test one dimensional stats for
  * @since 0.7.0
  * @author Ben Wilson, Databricks
  */
class OneDimStats(data: Array[Double]) {

  final private val SKEW_LOWER_BOUND = -1
  final private val SKEW_UPPER_BOUND = 1
  final private val KURTOSIS_LOWER_BOUND = 2.5
  final private val KURTOSIS_UPPER_BOUND = 3.5

  /**
    * Private method for calculating Kurtosis of the data set
    * @return Effective Kurtosis (k - 3)
    */
  private def calculateKurtosis: Double = new Kurtosis().evaluate(data)

  /**
    * Method for calculating Skewness of the data set
    * @return Skew value
    */
  private def calculateSkew: Double = new Skewness().evaluate(data)

  /**
    * Private method for calculating Variance of the data set
    * @return Variance value
    */
  private def calculateVariance: Double = new Variance().evaluate(data)

  /**
    * Private method for calculating Mean of the data set
    * @return Mean value
    */
  private def calculateMean: Double = new Mean().evaluate(data)

  /**
    * Private method for calculating the geometric mean and resetting in the event that the series is negative and
    * a geometric mean cannot be calculated
    * @return Geometric Mean value or 0.0 if invalid
    */
  private def calculateGeometricMean: Double = {
    val gm = new GeometricMean().evaluate(data)
    gm match {
      case x if x.isNaN => 0.0
      case _            => gm
    }
  }

  /**
    * Private method for calculating semi-variance (variance of values less than the mean) for the data set
    * @return Semi Variance value
    */
  private def calculateSemiVariance: Double = new SemiVariance().evaluate(data)

  /**
    * Private method for calculating the standard deviation for the data set
    * @return Standard Deviation value
    */
  private def calculateStandardDeviation: Double =
    new StandardDeviation().evaluate(data)

  /**
    * Private method for returning the statistics evaluator for the data series
    * @return SummaryStats payload
    */
  private def getSummaryStatistics: SummaryStats = {

    val stats = new SummaryStatistics()
    data.foreach(x => stats.addValue(x))

    SummaryStats(
      count = stats.getN,
      min = stats.getMin,
      max = stats.getMax,
      sum = stats.getSum,
      mean = stats.getMean,
      geometricMean = stats.getGeometricMean,
      variance = stats.getVariance,
      popVariance = stats.getPopulationVariance,
      secondMoment = stats.getSecondMoment,
      sumOfSquares = stats.getSumsq,
      stdDeviation = stats.getStandardDeviation,
      sumOfLogs = stats.getSumOfLogs
    )

  }

  /**
    * Helper method for comparing a distribution to the data series in order to get the p-value and the d-statistic
    * of difference from the 'shape' of the distribution.
    * @param test The payload consisting of test name (for pass-through) and the RealDistribution under test.
    * @return
    */
  private def compareKolmogorovSmirnov(
    test: DistributionTestPayload
  ): DistributionValidationData = {

    val p = TestUtils.kolmogorovSmirnovTest(test.distribution, data)
    val d = TestUtils.kolmogorovSmirnovStatistic(test.distribution, data)

    DistributionValidationData(test.testName, p, d)
  }

  /**
    * Method for comparing a data series to a set of standard distributions
    * @return DistributionTestResult, consisting of, at a top-level, the best fit
    *         based on p-value of similarity, the p-value, and the D-statistic.
    *         Also returned are all of the different distribution test results for the series (for performing
    *         distributed analytic roll-up on a distributed DataFrame's partitions)
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def compareToDistributions: DistributionTestResult = {

    val tests = Array(
      DistributionTestPayload("normal", new NormalDistribution(0, 1)),
      DistributionTestPayload("standardBetaPrime", new BetaDistribution(1, 1)),
      DistributionTestPayload("cauchy", new CauchyDistribution(0, 1)),
      DistributionTestPayload("standardChiSq", new ChiSquaredDistribution(1)),
      DistributionTestPayload("exponential", new ExponentialDistribution(1)),
      DistributionTestPayload("f", new FDistribution(1, 1)),
      DistributionTestPayload("erlang", new GammaDistribution(1, 2)),
      DistributionTestPayload("gumbel", new GumbelDistribution(1, 2)),
      DistributionTestPayload("laplace", new LaplaceDistribution(0, 1)),
      DistributionTestPayload("levy", new LevyDistribution(0, 1)),
      DistributionTestPayload("logistic", new LogisticDistribution(0, 1)),
      DistributionTestPayload("logNormal", new LogNormalDistribution(0, 1)),
      DistributionTestPayload("nakagami", new NakagamiDistribution(1, 1)),
      DistributionTestPayload(
        "pareto",
        new distribution.ParetoDistribution(1, 1)
      ),
      DistributionTestPayload("studentT", new TDistribution(1)),
      DistributionTestPayload("weibull", new WeibullDistribution(1, 1))
    )

    val allTests = tests.map(x => compareKolmogorovSmirnov(x))

    val bestFit = allTests.sortWith(_.pValue < _.pValue).head

    DistributionTestResult(
      bestDistributionFit = bestFit.test,
      bestDistributionPValue = bestFit.pValue,
      bestDistributionDStatistic = bestFit.dStatistic,
      allTests = allTests
    )

  }

  /**
    * Public method for executing analysis of all metrics
    * @return OneDimStatsData, containing all of the information for the analysis of the One Dimensional series of data.
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  def evaluate: OneDimStatsData = {

    val kurtosis = calculateKurtosis
    val skew = calculateSkew

    val kurtosisType = kurtosis match {
      case x if x <= KURTOSIS_LOWER_BOUND => "Platykurtic"
      case x if x > KURTOSIS_LOWER_BOUND && x < KURTOSIS_UPPER_BOUND =>
        "Mesokurtic"
      case x if x >= KURTOSIS_UPPER_BOUND => "Leptokurtic"
      case _                              => throw new IllegalArgumentException("Unsupported Kurtosis range")
    }

    val skewType = skew match {
      case x if x <= SKEW_LOWER_BOUND => "Asymmetrical Left Tailed"
      case x if x > SKEW_LOWER_BOUND && x < SKEW_UPPER_BOUND =>
        "Symmetric Normal"
      case x if x >= SKEW_UPPER_BOUND => "Asymmetrical Right Tailed"
      case _                          => throw new IllegalArgumentException("Unsupported Skew Range")
    }

    OneDimStatsData(
      calculateMean,
      calculateGeometricMean,
      calculateVariance,
      calculateSemiVariance,
      calculateStandardDeviation,
      skew,
      kurtosis,
      kurtosisType,
      skewType,
      getSummaryStatistics,
      compareToDistributions
    )

  }

}

/**
  * Companion Object
  */
object OneDimStats {

  def evaluate(data: Array[Double]): OneDimStatsData = {
    new OneDimStats(data).evaluate
  }

}
