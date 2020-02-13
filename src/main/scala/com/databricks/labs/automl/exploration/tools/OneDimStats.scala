package com.databricks.labs.automl.exploration.tools

import org.apache.commons.math3.stat.descriptive.moment._

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
    * Public method for executing analysis of all metrics
    * @return
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
      skewType
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
