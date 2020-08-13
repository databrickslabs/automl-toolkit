package com.databricks.labs.automl.exploration.analysis.shap.tools

case class ShapOutput(partition: Int,
                      rows: Long,
                      featureIndex: Int,
                      shapValue: Double)
    extends Serializable

private[analysis] case class MutatedVectors(
  referenceIncluded: org.apache.spark.ml.linalg.Vector,
  referenceExcluded: org.apache.spark.ml.linalg.Vector
) extends Serializable
