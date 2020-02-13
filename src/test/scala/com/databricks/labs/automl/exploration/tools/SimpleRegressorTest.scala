package com.databricks.labs.automl.exploration.tools

import com.databricks.labs.automl.AbstractUnitSpec

class SimpleRegressorTest extends AbstractUnitSpec {

  it should "give correct results with excessive outliers" in {

    val data1 = Seq(0.1, 0.02, 0.399, 0.4, 0.566, 0.6, 0.7, 0.8, 0.9, 1.0, 37.5,
      0.5, 17.8, 19.1)
    val data2 = Seq(2.0, 45.0, 6.0, 8.0, 17.0, 120.0, 104.0, 19.0, 18.0, 20.0,
      0.1, 99.9, 14.2, 63.2)

    val res = SimpleRegressor.calculate(data1, data2)

    assert(res.slope == -0.8302724558196052, "incorrect slope")
    assert(res.slopeStdErr == 1.0413727775143946, "incorrect slope stderr")
    assert(
      res.slopeConfidenceInterval == 2.2689563681145803,
      "incorrect slope CI"
    )
    assert(res.intercept == 43.08153224007564, "incorrect intercept")
    assert(
      res.interceptStdErr == 12.73014698229427,
      "incorrect intercept stderr"
    )
    assert(res.rSquared == 0.05030726305316163, "incorrect r-squared")
    assert(res.significance == 0.4407756266472016, "incorrect significance")
    assert(res.mse == 1768.2580059436523, "incorrect mean squared error")
    assert(res.rmse == 42.050659994150536, "incorrect rmse")
    assert(
      res.sumSquares == 1124.0210715333192,
      "incorrect sum of squares regression"
    )
    assert(
      res.totalSumSquares == 22343.117142857147,
      "incorrect total sum of squares"
    )
    assert(
      res.sumSquareError == 21219.096071323827,
      "incorrect sum square error"
    )
    assert(res.pairLength == 14L, "incorrect pair length")
    assert(
      res.pearsonR == -0.22429280651229463,
      "incorrect pearson R coefficient of correlation"
    )
    assert(
      res.crossProductSum == -1353.7978571428573,
      "incorrect cross product sum"
    )

  }

  it should "give correct results for generally linearly correlated series" in {

    val data1 =
      Seq(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 0.5, 2.1, 3.1)
    val data2 = Seq(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
      22.1, 24.9, 26.2, 28.2)

    val res = SimpleRegressor.calculate(data1, data2)

    assert(res.slope == 7.935469580626107, "incorrect slope")
    assert(res.slopeStdErr == 1.6828113994525227, "incorrect slope stderr")
    assert(
      res.slopeConfidenceInterval == 3.666531067037454,
      "incorrect slope CI"
    )
    assert(res.intercept == 7.6179858239810985, "incorrect intercept")
    assert(
      res.interceptStdErr == 2.1152611294308854,
      "incorrect intercept stderr"
    )
    assert(res.rSquared == 0.6495010445058065, "incorrect r-squared")
    assert(res.significance == 5.007338509579462E-4, "incorrect significance")
    assert(res.mse == 27.396166691277813, "incorrect mean squared error")
    assert(res.rmse == 5.234134760519432, "incorrect rmse")
    assert(
      res.sumSquares == 609.2059997046663,
      "incorrect sum of squares regression"
    )
    assert(res.totalSumSquares == 937.96, "incorrect total sum of squares")
    assert(
      res.sumSquareError == 328.75400029533375,
      "incorrect sum square error"
    )
    assert(res.pairLength == 14L, "incorrect pair length")
    assert(
      res.pearsonR == 0.8059162763623815,
      "incorrect pearson R coefficient of correlation"
    )
    assert(
      res.crossProductSum == 76.77000000000001,
      "incorrect cross product sum"
    )

  }

  it should "give correct results for perfectly linearly correlated series" in {

    val data1 =
      Seq(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 0.5, 2.1, 3.1)
    val data2 =
      Seq(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 0.5, 2.1, 3.1)

    val res = SimpleRegressor.calculate(data1, data2)

    assert(res.slope == 1.0, "incorrect slope")
    assert(res.slopeStdErr == 0.0, "incorrect slope stderr")
    assert(res.slopeConfidenceInterval == 0.0, "incorrect slope CI")
    assert(res.intercept == 0.0, "incorrect intercept")
    assert(res.interceptStdErr == 0.0, "incorrect intercept stderr")
    assert(res.rSquared == 1.0, "incorrect r-squared")
    assert(res.significance == 0.0, "incorrect significance")
    assert(res.mse == 0.0, "incorrect mean squared error")
    assert(res.rmse == 0.0, "incorrect rmse")
    assert(
      res.sumSquares == 9.674285714285716,
      "incorrect sum of squares regression"
    )
    assert(
      res.totalSumSquares == 9.674285714285716,
      "incorrect total sum of squares"
    )
    assert(res.sumSquareError == 0.0, "incorrect sum square error")
    assert(res.pairLength == 14L, "incorrect pair length")
    assert(
      res.pearsonR == 1.0,
      "incorrect pearson R coefficient of correlation"
    )
    assert(
      res.crossProductSum == 9.674285714285716,
      "incorrect cross product sum"
    )

  }

}
