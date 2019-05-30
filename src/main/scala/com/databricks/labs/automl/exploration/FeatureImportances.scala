package com.databricks.labs.automl.exploration

import com.databricks.labs.automl.sanitize.DataSanitizer
import org.apache.spark.sql.DataFrame

object CutoffTypes extends Enumeration {
  type CutoffTypes = Value
  val None, Threshold, Count = Value
}

case class FeatureImportanceConfig(
  labelCol: String,
  featuresCol: String,
  numericBoundaries: Map[String, (Double, Double)],
  stringBoundaries: Map[String, List[String]],
  scoringMetric: String,
  trainPortion: Double,
  trainSplitMethod: String,
  trainSplitChronologicalColumn: String,
  trainSplitChronlogicalRandomPercentage: Double,
  parallelism: Int,
  kFold: Int,
  seed: Long,
  optimizationStrategy: String,
  firstGenerationGenePool: Int,
  numberOfMutationGenerations: Int,
  numberOfMutationsPerGeneration: Int,
  numberOfParentsToRetain: Int,
  geneticMixing: Double,
  generationalMutationStrategy: String,
  mutationMagnitudeMode: String,
  fixedMutationValue: Int,
  earlyStoppingScore: Double,
  earlyStoppingFlag: Boolean,
  evolutionStrategy: String,
  continuousEvolutionMaxIterations: Int,
  continuousEvolutionStoppingScore: Double,
  continuousEvolutionParallelism: Int,
  continuousEvolutionMutationAggressiveness: Int,
  continuousEvolutionGeneticMixing: Double,
  continuousEvolutionRollingImprovementCount: Int,
  fieldsToIgnore: Array[String],
  numericFillStat: String,
  characterFillStat: String,
  modelSelectionDistinctThreshold: Int,
  modelType: String
)

class FeatureImportances(data: DataFrame, config: FeatureImportanceConfig)
    extends FeatureImportanceTools {

  // Pathing:
  // 1. DataPrep
  // 2. Tune model
  // 3. Extract FI based on model type (figure out how to get XGBoost importances!)
  // 4. Output everything that would be needed to then automatically tune and configure a run.

  def fillNaValues(): DataFrame = {

    val (cleanedData, fillMap, modelDetectedType) = new DataSanitizer(data)
      .setLabelCol(config.labelCol)
      .setFeatureCol(config.featuresCol)
      .setNumericFillStat(config.numericFillStat)
      .setCharacterFillStat(config.characterFillStat)
      .setModelSelectionDistinctThreshold(
        config.modelSelectionDistinctThreshold
      )
      .setFieldsToIgnoreInVector(config.fieldsToIgnore)
      .setParallelism(config.parallelism)
      .setFilterPrecision(0.01)
      .generateCleanData()

    cleanedData
  }

  def createFeatureVector(df: DataFrame): DataFrame = {}

}

trait FeatureImportanceTools {

  import CutoffTypes._

  private[exploration] def cutoffTypeEvaluator(value: String): CutoffTypes = {

    value.toLowerCase.replaceAll("\\s", "") match {
      case "none"  => None
      case "value" => Threshold
      case "count" => Count
      case _ =>
        throw new IllegalArgumentException(
          s"$value is not supported! Must be one of: 'none', 'value', or " +
            s"'count' "
        )
    }
  }

}
