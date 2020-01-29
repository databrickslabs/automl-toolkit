package com.databricks.labs.automl.exploration.tools

//TODO:
// Notes -> 1. Repartition to ensure that at max there are 5000 rows per partition
// 2. Then execute a mapPartitions through the Shapiro-Wilk test
// 3. take the highest value from the W test
// 4. calculate the Z statistic?

// TODO: Java code for shapiro-Wilk

/**
  * Shapiro-Wilk test for normality.
  *
  */
object ShapiroWilk extends ShapiroBase {

  /**
    * Calculates P-value for ShapiroWilk Test
    *
    * @param x
    * @return
    * @throws IllegalArgumentException
    */
  @throws[IllegalArgumentException]
  private def ShapiroWilkW(x: Array[Double]): ShapiroScoreData = {
    java.util.Arrays.sort(x)
    val n = x.length
    if (n < 3)
      throw new IllegalArgumentException(
        s"Count of elements to measure W is too small ($n) must be more than 3"
      )
    if (n > 5000) {
      throw new IllegalArgumentException(
        s"Count of elements to measure W is too large ($n) must be less than 5001"
      )
    }
    val nn2 = n / 2
    val a = new Array[Double](nn2 + 1)
    /* 1-based */
    /*
                ALGORITHM AS R94 APPL. STATIST. (1995) vol.44, no.4, 547-551.
                Calculates the Shapiro-Wilk W test and its significance level
     */
    val small = 1e-19
    /* polynomial coefficients */
    val g = Array(-2.273, 0.459)
    val c1 = Array(0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056)
    val c2 = Array(0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633)
    val c3 = Array(0.544, -0.39978, 0.025054, -6.714e-4)
    val c4 = Array(1.3822, -0.77857, 0.062767, -0.0020322)
    val c5 = Array(-1.5861, -0.31082, -0.083751, 0.0038915)
    val c6 = Array(-0.4803, -0.082676, 0.0030302)
    /* Local variables */
    var i = 0
    var j = 0
    var i1 = 0
    var ssassx = 0.0
    var summ2 = 0.0
    var ssumm2 = 0.0
    var gamma = 0.0
    var range = 0.0
    var a1 = 0.0
    var a2 = 0.0
    var an = 0.0
    var m = 0.0
    var s = 0.0
    var sa = 0.0
    var xi = 0.0
    var sx = 0.0
    var xx = 0.0
    var y = 0.0
    var w1 = 0.0
    var fac = 0.0
    var asa = 0.0
    var an25 = 0.0
    var ssa = 0.0
    var sax = 0.0
    var rsn = 0.0
    var ssx = 0.0
    var xsx = 0.0
    var pw = 1.0
    an = n.toDouble
    if (n == 3) a(1) = 0.70710678 /* = sqrt(1/2) */
    else {
      an25 = an + 0.25
      summ2 = 0.0
      i = 1
      while ({
        i <= nn2
      }) {
        a(i) = normalQuantile((i - 0.375) / an25, 0, 1) // p(X <= x),

        summ2 += a(i) * a(i)

        i += 1
      }
      summ2 *= 2.0
      ssumm2 = Math.sqrt(summ2)
      rsn = 1.0 / Math.sqrt(an)
      a1 = poly(c1, 6, rsn) - a(1) / ssumm2
      /* Normalize a[] */
      if (n > 5) {
        i1 = 3
        a2 = -a(2) / ssumm2 + poly(c2, 6, rsn)
        fac = Math.sqrt(
          (summ2 - 2.0 * (a(1) * a(1)) - 2.0 * (a(2) * a(2))) / (1.0 - 2.0 * (a1 * a1) - 2.0 * (a2 * a2))
        )
        a(2) = a2
      } else {
        i1 = 2
        fac = Math.sqrt((summ2 - 2.0 * (a(1) * a(1))) / (1.0 - 2.0 * (a1 * a1)))
      }
      a(1) = a1
      i = i1
      while ({
        i <= nn2
      }) {
        a(i) /= -fac

        i += 1
      }
    }
    range = x(n - 1) - x(0)
    if (range < small) {
      throw new IllegalArgumentException
    }
    /* Check for correct sort order on range - scaled X */
    xx = x(0) / range
    sx = xx
    sa = -a(1)
    i = 1
    j = n - 1
    while ({
      i < n
    }) {
      xi = x(i) / range
      if (xx - xi > small) {
        throw new IllegalArgumentException
      }
      sx += xi
      i += 1
      if (i != j) sa += sign(i - j) * a(Math.min(i, j))
      xx = xi

      j -= 1
    }
    // Calculate W statistic
    sa /= n
    sx /= n
    ssa = 0.0
    ssx = 0.0
    sax = 0.0
    i = 0
    j = n - 1
    while ({
      i < n
    }) {
      if (i != j) asa = sign(i - j) * a(1 + Math.min(i, j)) - sa
      else asa = -sa
      xsx = x(i) / range - sx
      ssa += asa * asa
      ssx += xsx * xsx
      sax += asa * xsx

      i += 1
      j -= 1
    }
    ssassx = Math.sqrt(ssa * ssx)
    w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx)
    val w = 1.0 - w1
    /* Calculate significance level for W */
    if (n == 3) {
      /* exact P value : */
      val pi6 = 1.90985931710274
      /* = 6/pi */
      val stqr = 1.04719755119660
      pw = pi6 * (Math.asin(Math.sqrt(w)) - stqr)
      if (pw < 0.0) pw = 0
      //return w;
      return ShapiroScoreData(w, 0.0, pw)
    }
    y = Math.log(w1)
    xx = Math.log(an)
    if (n <= 11) {
      gamma = poly(g, 2, an)
      if (y >= gamma) {
        pw = 1e-99
        return ShapiroScoreData(w, 0.0, pw)
      }
      y = -Math.log(gamma - y)
      m = poly(c3, 4, an)
      s = Math.exp(poly(c4, 4, an))
    } else {
      m = poly(c5, 4, xx)
      s = Math.exp(poly(c6, 3, xx))
    }
    val z = (y - m) / s
    pw = gaussCdf(z)
    ShapiroScoreData(w, z, pw)
  }

  /**
    * Tests the rejection of null Hypothesis for a particular confidence level
    *
    * @param data
    * @param aLevel
    * @return
    */
  def test(data: Array[Double], aLevel: Double): ShapiroInternalData = {
    var normalcyTest = false
    val swTest = ShapiroWilkW(data)
    val a = aLevel
    if (swTest.probability <= a || swTest.probability >= (1.0 - a))
      normalcyTest = true
    val normalcy = if (normalcyTest) NORMALCY_FALSE else NORMALCY_TRUE

    ShapiroInternalData(
      swTest.w,
      swTest.z,
      swTest.probability,
      normalcyTest,
      normalcy
    )
  }

}
// TODO: Stddev, Kurtosis, Variance, Sample Count, Mean
