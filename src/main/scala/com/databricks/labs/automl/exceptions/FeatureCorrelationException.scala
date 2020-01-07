package com.databricks.labs.automl.exceptions

final case class FeatureCorrelationException(
  private val originalFeatures: Array[String],
  private val filteredFeatures: Array[String],
  cause: Throwable = None.orNull
) extends RuntimeException(
      s"Feature Correlation Detection has filtered out every field from the feature candidates (feature count: " +
        s"${originalFeatures.length}).  Filtered Fields: ${filteredFeatures.mkString(", ")}",
      cause
    )
