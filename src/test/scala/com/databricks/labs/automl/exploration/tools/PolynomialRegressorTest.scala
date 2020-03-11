package com.databricks.labs.automl.exploration.tools

import com.databricks.labs.automl.AbstractUnitSpec
import org.apache.commons.math3.fitting.{
  PolynomialCurveFitter,
  WeightedObservedPoints
}

class PolynomialRegressorTest extends AbstractUnitSpec {

  //TODO: write actual tests here.

  it should "do something" in {

//    val data1 = Seq(0.1, 0.02, 0.399, 0.4, 0.566, 0.6, 0.7, 0.8, 0.9, 1.0, 37.5,
//      0.5, 17.8, 19.1)
//    val data2 = Seq(2.0, 45.0, 6.0, 8.0, 17.0, 120.0, 104.0, 19.0, 18.0, 20.0,
//      0.1, 99.9, 14.2, 63.2)

    val data1 = Seq(0.1, 0.02, 0.399, 0.4, 0.566, 0.6, 0.7, 0.8, 0.9, 1.0)
    val data2 = Seq(2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 11.0, 18.0, 4.0)

    val g = SimpleRegressor.calculate(data1, data2)
    println(g)

    // polynomial regressor....

    val theData = new WeightedObservedPoints()
    data1.zip(data2).foreach(x => theData.add(x._1, x._2))

    val fitter = PolynomialCurveFitter.create(2)

    val trial = fitter.fit(theData.toList)

    println(trial.mkString(","))

    val fitter2 = PolynomialCurveFitter.create(3)
    val trial2 = fitter2.fit(theData.toList)
    println(trial2.mkString(","))

    val fitter1 = PolynomialCurveFitter.create(1)
    val trial1 = fitter1.fit(theData.toList)
    println(trial1.mkString(","))

    val doIt = PolynomialRegressor.fit(data1, data2, 4)

    println(doIt)

    val mult =
      PolynomialRegressor.fitMultipleOrders(data1, data2, Array(1, 2, 3, 4, 5))
    mult.foreach(println)

  }

}
