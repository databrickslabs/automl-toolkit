package com.databricks.labs.automl.exploration.tools

import com.databricks.labs.automl.AbstractUnitSpec

class SimpleRegressorTest extends AbstractUnitSpec {

  it should "gimme results" in {

    val data1 = Seq(0.1, 0.02, 0.399, 0.4, 0.566, 0.6, 0.7, 0.8, 0.9, 1.0, 37.5,
      0.5, 17.8, 19.1)
    val data2 = Seq(2.0, 45.0, 6.0, 8.0, 17.0, 120.0, 104.0, 19.0, 18.0, 20.0,
      0.1, 99.9, 14.2, 63.2)

    val dat = IndexedSeq(data1, data2)

    var sum = 0.0
    for (pair <- dat) sum += pair(0)
    println(sum)

    val res = SimpleRegressor.calculate(dat)
    println(res)

  }

  it should "not blow up" in {

    val data1 =
      Seq(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 0.5, 2.1, 3.1)
    val data2 = Seq(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
      22.1, 24.9, 26.2, 28.2)

    println(data1.sum)
    println(data2.sum)
    val dat = IndexedSeq(data1, data2)

    var sum = 0.0
    for (pair <- dat) sum += pair(0)
    println(sum)

    val res = SimpleRegressor.calculate(dat)
    println(res)

  }

  it should "gimme perfect r2" in {

    val data1 =
      Seq(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 0.5, 2.1, 3.1)
    val data2 =
      Seq(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 0.5, 2.1, 3.1)

    println(data1.sum)
    println(data2.sum)
    val dat = IndexedSeq(data1, data2)

    var sum = 0.0
    for (pair <- dat) sum += pair(0)
    println(sum)

    val res = SimpleRegressor.calculate(dat)
    println(res)

  }

}
