package com.databricks.labs.automl.exploration.tools

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
    data.foreach(x => payload.add(x._1, x._2))

    payload
  }

  private def fitParameters(order: Int,
                            data: WeightedObservedPoints): Array[Double] = {

    //TODO: restrict the level here? order < 4? 5?

    val parameterFitter = PolynomialCurveFitter.create(order)

    parameterFitter.fit(data.toList)

  }

  private def linearEq(x: Double, m: Double, b: Double): Double = (m * x) + b

  private def secondOrderEq(x: Double,
                            a0: Double,
                            a1: Double,
                            a2: Double): Double = {
    (a2 * x * x) + (a1 * x) + a0
  }

  def fitSecondOrder(x: Seq[Double], y: Seq[Double]): Double = {

    val points = createObservations(x, y)
    val params = fitParameters(2, points)

    val zippedData = x.zip(y)

    val squaredError = zippedData.map(
      x =>
        math.pow(x._2 - secondOrderEq(x._1, params(0), params(1), params(2)), 2)
    )
    val sumSquareError = squaredError.sum
    val mse = sumSquareError / x.size
    val rmse = math.sqrt(mse)

    //TODO: regression slope error equation...SE of regression slope = sb1 = sqrt [ Σ(yi – ŷi)2 / (n – 2) ] / sqrt [ Σ(xi – x)2 ].

    //TODO: finish this to return the equation elements so that predictions can be made as well as all of the stats of the fit.
    rmse
  }

}
