package com.databricks.labs.automl.exploration.tools

import org.apache.commons.math3.stat.correlation.{
  Covariance,
  KendallsCorrelation,
  PearsonsCorrelation,
  SpearmansCorrelation
}
import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.apache.commons.math3.stat.inference.{TTest, TestUtils}

class PairedTesting(left: Array[Double],
                    right: Array[Double],
                    alpha: Double = 0.05) {

  /**
    * Private helper method for creating the PairedStatistics construct for calculating t tests
    * @return PairedSeq of SummaryStatistics Instances for both data series
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def createPairedStatistics: PairedSeq = {

    assert(
      left.length == right.length,
      s"Length of pairs is not equal. Left: ${left.length} Right: ${right.length}"
    )

    val leftStats = new SummaryStatistics()
    val rightStats = new SummaryStatistics()

    left.foreach(x => leftStats.addValue(x))
    right.foreach(x => rightStats.addValue(x))
    PairedSeq(leftStats, rightStats)

  }

  /**
    * Method for determining the t-test values for comparing if the mean values are equivalent between two sequences
    * of Doubles.
    * @return TTestData payload, consisting of the alpha that was used, the t-stat value, significance determination,
    *         significance p-value, and a judgement of equivalency (Y or N)
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def computePairedTTest: TTestData = {

    val pairData = createPairedStatistics

    val tStatisticValue = TestUtils.t(pairData.left, pairData.right)
    val tStatisticSignificance =
      TestUtils.pairedTTest(left, right, alpha)
    val tStatisticPValue = new TTest().pairedTTest(left, right)

    val equivalencyJudgement = tStatisticSignificance match {
      case x if x => "N"
      case _      => "Y"
    }

    TTestData(
      alpha = alpha,
      tStat = tStatisticValue,
      tTestSignificance = tStatisticSignificance,
      tTestPValue = tStatisticPValue,
      equivalencyJudgement = equivalencyJudgement.head
    )
  }

  /**
    * Equivalency tests for the distribution of data between two series.
    * @returns Payload of the equivalency p value, D statistic, and equivalency judgement between the two distributions
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def computeKolmogorovSmirnovTest = {

    val ksTestPValue =
      TestUtils.kolmogorovSmirnovTest(left, right)
    val ksTestDStatistic =
      TestUtils.kolmogorovSmirnovStatistic(left, right)
    val equivalency = ksTestPValue match {
      case x if x <= alpha => "Y"
      case _               => "N"
    }

    KSTestResult(ksTestPValue, ksTestDStatistic, equivalency.head)

  }

  /**
    * Method for calculating unbiased covariance between two data series
    * @return unbiased covariance score
    * @note https://commons.apache.org/proper/commons-math/apidocs/org/apache/commons/math4/stat/correlation/Covariance.html
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def computeCovariance: Double =
    new Covariance().covariance(left, right, false)

  /**
    * Method for calculating Pearson's product-moment correlation coefficient for two data series
    * @return Pearson's product-moment correlation coefficient
    * @note https://commons.apache.org/proper/commons-math/apidocs/org/apache/commons/math4/stat/correlation/PearsonsCorrelation.html
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def computePearsons: Double =
    new PearsonsCorrelation().correlation(left, right)

  /**
    * Method for calculating Spearman's Rank correlation for two data series using Natural Ranking
    * @return Spearman's rank correlation coefficient
    * @note https://commons.apache.org/proper/commons-math/apidocs/org/apache/commons/math4/stat/correlation/SpearmansCorrelation.html
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def computeSpearmans: Double =
    new SpearmansCorrelation().correlation(left, right)

  /**
    * Method for calculating Kendall's Tau-b Rank correlation for two data series
    * @return Kendall's Tau-b correlation coefficient
    * @note https://commons.apache.org/proper/commons-math/apidocs/org/apache/commons/math4/stat/correlation/KendallsCorrelation.html
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  private def computeKendallsTauRank: Double =
    new KendallsCorrelation().correlation(left, right)

  /**
    * Main execution method for getting the pair testing data for two data series.
    *   Performs equivalency correlation testing for:
    *     - Unbiased correlation testing
    *     - Pearson's correlation testing
    *     - Spearman's correlation testing
    *     - Kendall's correlation testing
    *   Computes a t-test for mean equivalency
    *   Computes distribution equivalency testing between the two series
    * @return Testing Payload of the statistical data
    * @since 0.7.0
    * @author Ben Wilson, Databricks
    */
  def execute: PairedTestResult = {

    val correlationChecks = CorrelationTestResult(
      computeCovariance,
      computePearsons,
      computeSpearmans,
      computeKendallsTauRank
    )

    PairedTestResult(
      correlationChecks,
      computePairedTTest,
      computeKolmogorovSmirnovTest
    )

  }

}

/**
  * Companion Object for Paired Testing
  */
object PairedTesting {

  def evaluate(left: Seq[Double],
               right: Seq[Double],
               alpha: Double): PairedTestResult = {
    new PairedTesting(left.toArray, right.toArray, alpha).execute
  }

  def evaluate(left: Array[Double],
               right: Array[Double],
               alpha: Double): PairedTestResult = {
    new PairedTesting(left, right, alpha).execute
  }

}
