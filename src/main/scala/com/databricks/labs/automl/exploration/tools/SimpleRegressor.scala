package com.databricks.labs.automl.exploration.tools

import org.apache.commons.math3.stat.regression.SimpleRegression

/**
  *
  */
object SimpleRegressor {

  private def createMatchedPairs(x: Seq[Double],
                                 y: Seq[Double]): Array[Array[Double]] = {

    x.zip(y).map(x => Array(x._1, x._2)).toArray

  }

  def calculate(x: Seq[Double], y: Seq[Double]): SimpleRegressorResult = {

    val pairs = createMatchedPairs(x, y)

    val r = new SimpleRegression()

    r.addData(pairs)

    SimpleRegressorResult(
      r.getSlope,
      r.getSlopeStdErr,
      r.getSlopeConfidenceInterval,
      r.getIntercept,
      r.getInterceptStdErr,
      r.getRSquare,
      r.getSignificance,
      r.getMeanSquareError,
      math.sqrt(r.getMeanSquareError),
      r.getRegressionSumSquares,
      r.getTotalSumSquares,
      r.getSumSquaredErrors,
      r.getN,
      r.getR,
      r.getSumOfCrossProducts
    )

  }

}
