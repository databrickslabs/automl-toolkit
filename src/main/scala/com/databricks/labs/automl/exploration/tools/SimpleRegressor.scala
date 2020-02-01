package com.databricks.labs.automl.exploration.tools

import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration

/**
  *
  */
object SimpleRegressor {

  private def summation(data: IndexedSeq[Seq[Double]],
                        idx: Int): Future[Double] = {
    Future { data.foldLeft(0.0) { case (acc, x) => acc + x(idx) } }
  }
  private def summedSquares(data: IndexedSeq[Seq[Double]],
                            idx: Int): Future[Double] = {
    Future { data.foldLeft(0.0) { case (acc, x) => acc + math.pow(x(idx), 2) } }
  }

  private def pairProduct(data: IndexedSeq[Seq[Double]]): Future[Double] = {
    Future { data.foldLeft(0.0) { case (acc, x) => acc + (x(0) * x(1)) } }
  }

  private def calculateResiduals(data: IndexedSeq[Seq[Double]],
                                 barData: RegressionBarData,
                                 xBar: Double,
                                 yBar: Double): RegressionResidualData = {

    var ssr = 0.0
    var rss = 0.0
    for (i <- data(0).indices) {
      val fit = ((barData.xyBar / barData.xBar) * data(0)(i)) + (yBar - (barData.xyBar / barData.xBar) * xBar)
      ssr += ((fit - yBar) * (fit - yBar))
      rss += ((fit - data(1)(i)) * (fit - data(1)(i)))
    }

    RegressionResidualData(ssr, rss)
  }

  private def calculateBarData(data: IndexedSeq[Seq[Double]],
                               xBar: Double,
                               yBar: Double): RegressionBarData = {

    var xxBar = 0.0
    var yyBar = 0.0
    var xyBar = 0.0
    for (i <- data(0).indices) {
      xxBar += ((data(0)(i) - xBar) * (data(0)(i) - xBar))
      yyBar += ((data(1)(i) - yBar) * (data(1)(i) - yBar))
      xyBar += ((data(0)(i) - xBar) * (data(1)(i) - yBar))
    }

    RegressionBarData(xxBar, yyBar, xyBar)

  }

  def calculate(pairedData: IndexedSeq[Seq[Double]]): SimpleRegressorResult = {

    val n = pairedData(0).size

    require(
      pairedData(0).size == pairedData(1).size,
      s"length of paired data Seq do not match."
    )

    val evaluation = for {

      sumX <- summation(pairedData, 0)
      sumY <- summation(pairedData, 1)
      sumSqX <- summedSquares(pairedData, 0)
      sumSqY <- summedSquares(pairedData, 1)
      sumProduct <- pairProduct(pairedData)

    } yield RegressionInternal(sumX, sumY, sumSqX, sumSqY, sumProduct)

    val result = Await.result(evaluation, Duration.Inf)

    val xBar = result.sumX / n
    val yBar = result.sumY / n

    val barData = calculateBarData(pairedData, xBar, yBar)

    val residuals = calculateResiduals(pairedData, barData, xBar, yBar)

    val slope = barData.xyBar / barData.xBar
    val intercept = yBar - (slope * xBar)
    val degreesFreedom = pairedData(0).size - 2
    val r2 = residuals.ssr / barData.yBar
    val svar = residuals.rss / degreesFreedom
    val svar1 = svar / barData.xBar
    val svar0 = (svar / pairedData(0).size) + (xBar * xBar * svar1)
    val interceptStdErr = math.sqrt(svar0)
    val slopeStdErr = math.sqrt(svar1)

    SimpleRegressorResult(slope, slopeStdErr, intercept, interceptStdErr, r2)

  }

}
