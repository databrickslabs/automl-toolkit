package com.databricks.labs.automl.exploration.tools

case class OneDimStatsData(mean: Double,
                           geomMean: Double,
                           variance: Double,
                           semiVariance: Double,
                           stddev: Double,
                           skew: Double,
                           kurtosis: Double,
                           kurtosisType: String,
                           skewType: String)

case class ShapiroScoreData(w: Double, z: Double, probability: Double)

case class ShapiroInternalData(w: Double,
                               z: Double,
                               probability: Double,
                               normalcyTest: Boolean,
                               normalcy: String)
