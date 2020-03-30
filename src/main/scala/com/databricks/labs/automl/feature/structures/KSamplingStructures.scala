package com.databricks.labs.automl.feature.structures

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types._

trait KSamplingDefaults {

  def defaultFeaturesCol = "features"
  def defaultLabelCol = "label"
  def defaultSyntheticCol = "synthetic"
  def defaultFieldsToIgnore: Array[String] = Array[String]()
  def defaultKGroups = 25
  def defaultKMeansMaxIter = 100
  def defaultKMeansTolerance = 1E-6
  def defaultKMeansDistanceMeasurement = "euclidean"
  def defaultKMeansSeed = 42L
  def defaultKMeansPredictionCol = "kGroups"
  def defaultHashTables = 10
  def defaultLSHSeed = 42L
  def defaultLSHOutputCol = "hashes"
  def defaultQuorumCount = 7
  def defaultMinimumVectorCountToMutate = 1
  def defaultVectorMutationMethod = "random"
  def defaultMutationMode = "weighted"
  def defaultMutationValue = 0.5

  def defaultFill: Map[DataType, Any] =
    Map(
      DoubleType -> 0.0,
      IntegerType -> 0,
      StringType -> "hodor",
      ShortType -> 0,
      LongType -> 0L,
      FloatType -> 0.0,
      BooleanType -> true,
      TimestampType -> "1980-01-08T08:03:52.0",
      DateType -> "1980-06-01",
      BinaryType -> Array(0, 1, 1, 0)
    )
}

case class CentroidVectors(vector: Vector, kGroup: Int)

case class KSamplingConfiguration(featuresCol: String,
                                  labelCol: String,
                                  syntheticCol: String,
                                  fieldsToIgnore: Array[String],
                                  kGroups: Int,
                                  kMeansMaxIter: Int,
                                  kMeansTolerance: Double,
                                  kMeansDistanceMeasurement: String,
                                  kMeansSeed: Long,
                                  kMeansPredictionCol: String,
                                  lshHashTables: Int,
                                  lshSeed: Long,
                                  lshOutputCol: String,
                                  quorumCount: Int,
                                  minimumVectorCountToMutate: Int,
                                  vectorMutationMethod: String,
                                  mutationMode: String,
                                  mutationValue: Double)

case class SchemaMapping(fieldName: String,
                         originalFieldIndex: Int,
                         dfType: DataType,
                         scalaType: String)

case class StructMapping(field: StructField, idx: Int)

case class RowMapping(fieldName: String, idx: Int)

case class SchemaDefinitions(fullSchema: Array[SchemaMapping],
                             features: Array[RowMapping])

case class RowGenerationConfig(labelValue: Double, targetCount: Int)

case class CardinalityPayload(labelValue: Double, labelCounts: Int)
