package com.databricks.labs.automl.exploration.tools

import com.databricks.labs.automl.AbstractUnitSpec

class ShapiroWilkTest extends AbstractUnitSpec {

  final val ALPHA = 0.05

  final val NORMAL_DATA =
    Array(-10.0, -5.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 5.0, 10.0)
  final val LEFT_SKEW_DATA =
    Array(-1000.0, -500.0, -100.0, -50.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
  final val RIGHT_SKEW_DATA =
    Array(0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 10.0, 100.0, 1000.0, 10000.0)
  final val ZERO_RANGE_DATA = Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
  final val LOW_RANGE_DATA =
    Array(1.1111111e-15, 1.11111e-15, 1.11111e-15, 1.11111e-15, 1.11111e-15,
      1.111111e-15)
  final val SMALL_DATA = Array(0.0, 1.0)

  it should "throw an exception for extremely low range data" in {
    intercept[IllegalArgumentException] {
      ShapiroWilk.test(LOW_RANGE_DATA, ALPHA)
    }
  }

  it should "throw an exception for zero range data" in {
    intercept[IllegalArgumentException] {
      ShapiroWilk.test(ZERO_RANGE_DATA, ALPHA)
    }
  }

  it should "throw an exception for small data" in {
    intercept[IllegalArgumentException] {
      ShapiroWilk.test(SMALL_DATA, ALPHA)
    }
  }

  it should "return the correct result for normal data for Shapiro-Wilk" in {

    val EXPECTED_W = 0.9059018777993929
    val EXPECTED_Z = 0.7792008630260997
    val EXPECTED_PROB = 0.7820692990441541
    val EXPECTED_NORMALCY = false
    val EXPECTED_DECISION = "Y"

    val result = ShapiroWilk.test(NORMAL_DATA, ALPHA)

    assert(
      result.w == EXPECTED_W,
      "failed to calculate the correct W value for normal distribution"
    )
    assert(
      result.z == EXPECTED_Z,
      "failed to calculate Z score properly for normal distribution"
    )
    assert(
      result.probability == EXPECTED_PROB,
      "failed to calculate probability properly for normal distribution"
    )
    assert(
      result.normalcyTest == EXPECTED_NORMALCY,
      "failed to determine normalcy test for normal distribution"
    )
    assert(
      result.normalcy == EXPECTED_DECISION,
      "failed to define normalcy decision correctly for normal distribution"
    )
  }

  it should "return the correct result for left skewed data for Shapiro-Wilk" in {

    val EXPECTED_W = 0.5926784836883106
    val EXPECTED_Z = 3.914789120443548
    val EXPECTED_PROB = 0.9999547585887872
    val EXPECTED_NORMALCY = true
    val EXPECTED_DECISION = "N"

    val result = ShapiroWilk.test(LEFT_SKEW_DATA, ALPHA)

    assert(
      result.w == EXPECTED_W,
      "failed to calculate the correct W value for normal distribution"
    )
    assert(
      result.z == EXPECTED_Z,
      "failed to calculate Z score properly for normal distribution"
    )
    assert(
      result.probability == EXPECTED_PROB,
      "failed to calculate probability properly for normal distribution"
    )
    assert(
      result.normalcyTest == EXPECTED_NORMALCY,
      "failed to determine normalcy test for normal distribution"
    )
    assert(
      result.normalcy == EXPECTED_DECISION,
      "failed to define normalcy decision correctly for normal distribution"
    )
  }

  it should "return the correct result for right skewed data for Shapiro-Wilk" in {

    val EXPECTED_W = 0.41812749335736665
    val EXPECTED_Z = 4.933291170618088
    val EXPECTED_PROB = 0.9999995955990474
    val EXPECTED_NORMALCY = true
    val EXPECTED_DECISION = "N"

    val result = ShapiroWilk.test(RIGHT_SKEW_DATA, ALPHA)

    assert(
      result.w == EXPECTED_W,
      "failed to calculate the correct W value for normal distribution"
    )
    assert(
      result.z == EXPECTED_Z,
      "failed to calculate Z score properly for normal distribution"
    )
    assert(
      result.probability == EXPECTED_PROB,
      "failed to calculate probability properly for normal distribution"
    )
    assert(
      result.normalcyTest == EXPECTED_NORMALCY,
      "failed to determine normalcy test for normal distribution"
    )
    assert(
      result.normalcy == EXPECTED_DECISION,
      "failed to define normalcy decision correctly for normal distribution"
    )
  }

}
