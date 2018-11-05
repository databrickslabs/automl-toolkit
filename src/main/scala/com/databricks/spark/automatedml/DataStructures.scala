package com.databricks.spark.automatedml




case class PearsonPayload(fieldName: String,
                          pvalue: Double,
                          degreesFreedom: Int,
                          pearsonStat: Double)

case class FilterData(
                       field: String,
                       uniqueValues: Long
                     )

case class RandomForestConfig(numTrees: Int,
                              impurity: String,
                              maxBins: Int,
                              maxDepth: Int,
                              minInfoGain: Double,
                              subSamplingRate: Double,
                              featureSubsetStrategy: String
                             )
case class ModelsWithResults(modelHyperParams: RandomForestConfig,
                             model: Any,
                             score: Double,
                             evalMetrics: Map[String, Double],
                             generation: Int)


case class StaticModelConfig(labelColumn: String, featuresColumn: String)

sealed trait ModelType[A,B]
final case class ClassiferType[A,B](a: A) extends ModelType[A,B]
final case class RegressorType[A,B](b: B) extends ModelType[A,B]


