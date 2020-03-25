package com.databricks.labs.automl.exploration.tools

import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}

class OneDimStatsTest extends AbstractUnitSpec {

  final val ALPHA = 0.05

  final val PERFECT_NORMAL_DATA = Array(2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0,
    5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0,
    8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 11.0, 11.0, 12.0)

  it should "calculate stats correctly for a decaying right skewed distribution" in {

    val EXPECTED_MEAN = -0.6548573547866678
    val EXPECTED_GEOM = 0.0
    val EXPECTED_VARIANCE = 9.950225427633128E-5
    val EXPECTED_SEMI_VARIANCE = 2.4970686878096203E-5
    val EXPECTED_STDDEV = 0.009975081667652215
    val EXPECTED_SKEW = 7.75779394110764
    val EXPECTED_KURTOSIS = 132.06335325374056
    val EXPECTED_KURTOSIS_TYPE = "Leptokurtic"
    val EXPECTED_SKEW_TYPE = "Asymmetrical Right Tailed"

    val data = DiscreteTestDataGenerator.generateDecayArray(500).map(_ * -1)

    val result = OneDimStats.evaluate(data)
    println(result)
    assert(result.mean == EXPECTED_MEAN, "mean is not correct")
    assert(result.geomMean == EXPECTED_GEOM, "geometric mean is not correct")
    assert(result.variance == EXPECTED_VARIANCE, "variance is not correct")
    assert(
      result.semiVariance == EXPECTED_SEMI_VARIANCE,
      "semi variance is not correct"
    )
    assert(result.stddev == EXPECTED_STDDEV, "stddev is not correct")
    assert(result.skew == EXPECTED_SKEW, "skew is not correct")
    assert(result.kurtosis == EXPECTED_KURTOSIS, "kurtosis is not correct")
    assert(
      result.kurtosisType == EXPECTED_KURTOSIS_TYPE,
      "kurtosis type is not correct"
    )
    assert(result.skewType == EXPECTED_SKEW_TYPE, "skew type is not correct")

  }

  it should "calculate stats correctly for a chaotic distribution" in {

    val EXPECTED_MEAN = 0.6475713954442238
    val EXPECTED_GEOM = 0.6088821996450652
    val EXPECTED_VARIANCE = 0.04587715724784365
    val EXPECTED_SEMI_VARIANCE = 0.024330243806613878
    val EXPECTED_STDDEV = 0.21418953580379144
    val EXPECTED_SKEW = -0.17336511627873177
    val EXPECTED_KURTOSIS = -1.698405967752837
    val EXPECTED_KURTOSIS_TYPE = "Platykurtic"
    val EXPECTED_SKEW_TYPE = "Symmetric Normal"

    val data = DiscreteTestDataGenerator.generateChaoticArray(500)

    val result = OneDimStats.evaluate(data)

    assert(result.mean == EXPECTED_MEAN, "mean is not correct")
    assert(result.geomMean == EXPECTED_GEOM, "geometric mean is not correct")
    assert(result.variance == EXPECTED_VARIANCE, "variance is not correct")
    assert(
      result.semiVariance == EXPECTED_SEMI_VARIANCE,
      "semi variance is not correct"
    )
    assert(result.stddev == EXPECTED_STDDEV, "stddev is not correct")
    assert(result.skew == EXPECTED_SKEW, "skew is not correct")
    assert(result.kurtosis == EXPECTED_KURTOSIS, "kurtosis is not correct")
    assert(
      result.kurtosisType == EXPECTED_KURTOSIS_TYPE,
      "kurtosis type is not correct"
    )
    assert(result.skewType == EXPECTED_SKEW_TYPE, "skew type is not correct")

  }

  it should "calculate stats correctly for a decaying left tailed distribution" in {

    val EXPECTED_MEAN = 0.6535971281474307
    val EXPECTED_GEOM = 0.6531836300290403
    val EXPECTED_VARIANCE = 4.995262976704292E-4
    val EXPECTED_SEMI_VARIANCE = 3.6385129297841144E-4
    val EXPECTED_STDDEV = 0.022350084958908528
    val EXPECTED_SKEW = -3.360258460470165
    val EXPECTED_KURTOSIS = 24.457852041324198
    val EXPECTED_KURTOSIS_TYPE = "Leptokurtic"
    val EXPECTED_SKEW_TYPE = "Asymmetrical Left Tailed"

    val data = DiscreteTestDataGenerator.generateDecayArray(100)

    val result = OneDimStats.evaluate(data)

    assert(result.mean == EXPECTED_MEAN, "mean is not correct")
    assert(result.geomMean == EXPECTED_GEOM, "geometric mean is not correct")
    assert(result.variance == EXPECTED_VARIANCE, "variance is not correct")
    assert(
      result.semiVariance == EXPECTED_SEMI_VARIANCE,
      "semi variance is not correct"
    )
    assert(result.stddev == EXPECTED_STDDEV, "stddev is not correct")
    assert(result.skew == EXPECTED_SKEW, "skew is not correct")
    assert(result.kurtosis == EXPECTED_KURTOSIS, "kurtosis is not correct")
    assert(
      result.kurtosisType == EXPECTED_KURTOSIS_TYPE,
      "kurtosis type is not correct"
    )
    assert(result.skewType == EXPECTED_SKEW_TYPE, "skew type is not correct")

  }

  it should "calculate stats correctly for Normal Distributions" in {

    val EXPECTED_MEAN = 7.0
    val EXPECTED_GEOM = 6.52006260856204
    val EXPECTED_VARIANCE = 6.0
    val EXPECTED_SEMI_VARIANCE = 3.0
    val EXPECTED_STDDEV = 2.449489742783178
    val EXPECTED_SKEW = 0.0
    val EXPECTED_KURTOSIS = -0.5449197860962554
    val EXPECTED_KURTOSIS_TYPE = "Platykurtic"
    val EXPECTED_SKEW_TYPE = "Symmetric Normal"

    val result = OneDimStats.evaluate(PERFECT_NORMAL_DATA)

    assert(result.mean == EXPECTED_MEAN, "mean is not correct")
    assert(result.geomMean == EXPECTED_GEOM, "geometric mean is not correct")
    assert(result.variance == EXPECTED_VARIANCE, "variance is not correct")
    assert(
      result.semiVariance == EXPECTED_SEMI_VARIANCE,
      "semi variance is not correct"
    )
    assert(result.stddev == EXPECTED_STDDEV, "stddev is not correct")
    assert(result.skew == EXPECTED_SKEW, "skew is not correct")
    assert(result.kurtosis == EXPECTED_KURTOSIS, "kurtosis is not correct")
    assert(
      result.kurtosisType == EXPECTED_KURTOSIS_TYPE,
      "kurtosis type is not correct"
    )
    assert(result.skewType == EXPECTED_SKEW_TYPE, "skew type is not correct")

  }

}
