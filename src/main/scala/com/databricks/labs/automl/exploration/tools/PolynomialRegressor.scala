package com.databricks.labs.automl.exploration.tools

import org.apache.commons.math3.analysis.polynomials
import org.apache.commons.math3.analysis.polynomials.PolynomialFunction
import org.apache.commons.math3.analysis.solvers.LaguerreSolver
import org.apache.commons.math3.fitting
import org.apache.commons.math3.fitting.{
  PolynomialCurveFitter,
  WeightedObservedPoints
}

object PolynomialRegressor {

  private def createObservations(x: Seq[Double],
                                 y: Seq[Double]): WeightedObservedPoints = {

    val data = x.zip(y)
    val payload = new fitting.WeightedObservedPoints()

    data.foreach { case (x, y) => payload.add(x, y) }

    payload
  }

  private def fitParameters(order: Int,
                            data: WeightedObservedPoints): Array[Double] = {

    val parameterFitter = PolynomialCurveFitter.create(order)

    parameterFitter.fit(data.toList)

  }

  def getRoot(params: Array[Double]): Double = {

    val polynomial = new polynomials.PolynomialFunction(params)
    val solver = new LaguerreSolver()
    solver.solve(100, polynomial, -1000, 1000)
  }

  private def calculateFit(x: Double, poly: PolynomialFunction): Double = {
    poly.value(x)
  }

  private def calculateSSR(data: Seq[(Double, Double)],
                           predictions: Seq[Double]): Double = {

    data.map(_._2).zip(predictions).foldLeft(0.0) {
      case (acc, i) =>
        acc + math.pow(i._1 - i._2, 2)
    }
  }

  private def calculateSSE(predictions: Seq[Double],
                           meanActual: Double): Double = {
    predictions.foldLeft(0.0) {
      case (acc, i) => acc + math.pow(i - meanActual, 2)
    }
  }

  private def calculateSST(data: Seq[(Double, Double)],
                           meanActual: Double): Double = {
    data.map(_._2).foldLeft(0.0) {
      case (acc, i) => acc + math.pow(i - meanActual, 2)
    }
  }

  private def calculateR2(ssr: Double, sst: Double): Double = {
    1.0 - (ssr / sst)
  }

  def fit(x: Seq[Double],
          y: Seq[Double],
          order: Int): PolynomialRegressorResult = {

    val points = createObservations(x, y)
    val params = fitParameters(order, points)
    val polynomial = new PolynomialFunction(params)

    val zippedData = x.zip(y)

    val predictions = x.map(a => calculateFit(a, polynomial))

    val ssr = calculateSSR(zippedData, predictions)
    val sse = calculateSSE(predictions, y.sum / y.length)
    val sst = calculateSST(zippedData, y.sum / y.length)
    val mse = ssr / (x.size - order)
    val rmse = math.sqrt(mse)
    val r2 = calculateR2(ssr, sst)

    PolynomialRegressorResult(order, polynomial, ssr, sse, sst, mse, rmse, r2)

  }

  def fitMultipleOrders(
    x: Seq[Double],
    y: Seq[Double],
    orders: Array[Int]
  ): Array[PolynomialRegressorResult] = {

    orders.map(o => fit(x, y, o))

  }

}
