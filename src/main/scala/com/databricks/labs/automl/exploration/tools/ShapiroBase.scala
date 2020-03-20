package com.databricks.labs.automl.exploration.tools

trait ShapiroBase {

  final val C1 =
    Array(0.0, 0.221157E0, -0.147981E0, -0.207119E1, 0.4434685E1, -0.2706056E1)
  final val C2 = Array(0.0E0, 0.42981E-1, -0.293762E0, -0.1752461E1,
    0.5682633E1, -0.3582633E1)
  final val C3 = Array(0.5440E0, -0.39978E0, 0.25054E-1, -0.6714E-3)
  final val C4 =
    Array(0.13822E1, -0.77857E0, 0.62767E-1, -0.20322E-2)
  final val C5 =
    Array(Double.NaN, -0.15861E1, -0.31082E0, -0.83751E-1, 0.38915E-2)
  final val C6 = Array(-0.4803E0, -0.82676E-1, 0.30302E-2)
  final val C7 = Array(0.164E0, 0.533E0)
  final val C8 = Array(0.1736E0, 0.315E0)
  final val C9 = Array(0.256E0, -0.635E-2)
  final val G = Array(-0.2273E1, 0.459E0)
  final val Z90 = 0.12816E1
  final val Z95 = 0.16449E1
  final val Z99 = 0.23263E1
  final val ZM = 0.17509E1
  final val ZSS = 0.56268E0
  final val BF1 = 0.8378E0
  final val XX90 = 0.556E0
  final val XX95 = 0.622E0
  final val SQRTH = 0.70711E0
  final val TH = 0.375E0
  final val SMALL = 1E-19
  final val PI6 = 0.1909859E1
  final val STQR = 0.1047198E1
  final val UPPER = true

  final val NORMALCY_TRUE = "Y"
  final val NORMALCY_FALSE = "N"

  /**
    * Compute the quantile function for the normal distribution. For small to moderate probabilities, algorithm referenced
    * below is used to obtain an initial approximation which is polished with a final Newton step. For very large arguments, an algorithm of Wichura is used.
    * Used by ShapiroWilk Test
    * Ported by Javascript implementation found at https://raw.github.com/rniwa/js-shapiro-wilk/master/shapiro-wilk.js
    * Originally ported from http://svn.r-project.org/R/trunk/src/nmath/qnorm.c
    *
    * @param p
    * @param mu
    * @param sigma
    * @return
    */
  def normalQuantile(p: Double, mu: Double, sigma: Double): Double = { // The inverse of cdf.
    if (sigma < 0)
      throw new IllegalArgumentException(
        "The sigma parameter must be positive."
      )
    else if (sigma == 0) return mu
    var r = .0
    var `val` = .0
    val q = p - 0.5
    if (0.075 <= p && p <= 0.925) {
      r = 0.180625 - q * q
      `val` = q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853)
        * r + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427)
        * r + 133.14166789178437745) * r + 3.387132872796366608) /
        (((((((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061)
          * r + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083)
          * r + 42.313330701600911252) * r + 1)
    } else {
      /* closer than 0.075 from {0,1} boundary */ /* r = min(p, 1-p) < 0.075 */
      if (q > 0) r = 1 - p
      else r = p /* = R_DT_Iv(p) ^=  p */
      r = Math.sqrt(-Math.log(r)) /* r = sqrt(-log(r))  <==>  min(p, 1-p) = exp( - r^2 ) */
      if (r <= 5.0) {
        /* <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11 */
        r += -1.6
        `val` = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177)
          * r + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055)
          * r + 4.6303378461565452959) * r + 1.42343711074968357734) /
          (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966)
            * r + 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494)
            * r + 2.05319162663775882187) * r + 1.0)
      } else {
        /* very close to  0 or 1 */
        r += -5.0
        `val` = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386)
          * r + 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358)
          * r + 5.4637849111641143699) * r + 6.6579046435011037772) /
          (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5)
            * r + 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531)
            * r + 0.59983220655588793769) * r + 1.0)
      }
      if (q < 0.0) `val` = -`val`
    }
    mu + sigma * `val`
  }

  def gaussCdf(z: Double): Double = { // input = z-value (-inf to +inf)

    // ACM Algorithm #209
    var y = 0.0 // 209 scratch variable
    var p = 0.0 // result. called ‘z’ in 209
    var w = 0.0
    if (z == 0.0) p = 0.0
    else {
      y = Math.abs(z) / 2.0
      if (y >= 3.0) p = 1.0
      else if (y < 1.0) {
        w = y * y
        p = ((((((((0.000124818987 * w - 0.001075204047) * w + 0.005198775019) * w - 0.019198292004)
          * w + 0.059054035642) * w - 0.151968751364) * w + 0.319152932694) * w - 0.531923007300)
          * w + 0.797884560593) * y * 2.0
      } else {
        y = y - 2.0
        p = (((((((((((((-0.000045255659 * y + 0.000152529290) * y - 0.000019538132)
          * y - 0.000676904986) * y + 0.001390604284) * y - 0.000794620820) * y - 0.002034254874)
          * y + 0.006549791214) * y - 0.010557625006) * y + 0.011630447319) * y - 0.009279453341)
          * y + 0.005353579108) * y - 0.002141268741) * y + 0.000535310849) * y + 0.999936657524
      }
    }
    if (z > 0.0) return (p + 1.0) / 2.0
    (1.0 - p) / 2.0
  }

  /**
    * Used internally by ShapiroWilkW().
    *
    * @param cc
    * @param nord
    * @param x
    * @return
    */
  def poly(cc: Array[Double], nord: Int, x: Double): Double = {
    /* Algorithm AS 181.2    Appl. Statist.    (1982) Vol. 31, No. 2
           Calculates the algebraic polynomial of order nord-1 with array of coefficients cc.
           Zero order coefficient is cc(1) = cc[0] */
    var ret_val = cc(0)
    if (nord > 1) {
      var p = x * cc(nord - 1)
      for (j <- nord - 2 until 0 by -1) {
        p = (p + cc(j)) * x
      }
      ret_val += p
    }
    ret_val
  }

  /**
    * Used internally by ShapiroWilkW()
    *
    * @param x
    * @return
    */
  def sign(x: Double): Int = {
    if (x == 0) return 0
    if (x > 0) 1
    else -1
  }

}
