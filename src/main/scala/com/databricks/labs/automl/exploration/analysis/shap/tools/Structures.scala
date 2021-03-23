package com.databricks.labs.automl.exploration.analysis.shap.tools
import org.apache.spark.ml.linalg.Vector

//case class ShapOutput(partition: Int,
//                      rows: Long,
//                      featureIndex: Int,
//                      shapValue: Double)
//    extends Serializable


case class ShapResult(features: Vector,
                      shapleyValues: Array[Double],
                      shapleyErrorEstimate: Array[Double])
      extends Serializable

case class EfficiencyResult(shapleySum: Double, predictionDelta: Double, relativeDiff: Double)

private[analysis] case class ShapVal(value: Double,
                   stdErr: Double)
      extends Serializable


private[analysis] case class MutatedVectors(
  referenceIncluded: org.apache.spark.ml.linalg.Vector,
  referenceExcluded: org.apache.spark.ml.linalg.Vector
) extends Serializable

private[analysis] case class VarianceAccumulator(sum: Double,
                                                 sumSquare: Double,
                                                 n: Int) {
  def mean: Double = sum / n.toDouble

  def variance: Double = (sumSquare - sum * sum / n) / (n - 1)

  def standardError: Double = math.sqrt(variance / n)
}


