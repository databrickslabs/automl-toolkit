package com.databricks.labs.automl.feature

trait SyntheticFeatureBase extends KSamplingBase {

  final val allowableLabelBalanceModes: List[String] =
    List("match", "percentage", "target")

  def defaultCardinalityThreshold: Int = 20
  def defaultLabelBalanceMode: String = "match"
  def defaultNumericRatio: Double = 0.2
  def defaultNumericTarget: Int = 500
}
