package com.databricks.spark.automatedml.reports

import com.databricks.spark.automatedml.model.RandomForestTuner
import com.databricks.spark.automatedml.params.{MainConfig, RandomForestModelsWithResults}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.DataFrame

class RandomForestFeatureImportance(data: DataFrame, featConfig: MainConfig, modelType: String)
  extends ReportingTools {

  def runFeatureImportances(fields: Array[String]): (RandomForestModelsWithResults, DataFrame) = {

    val (modelResults, modelStats) = new RandomForestTuner(data, modelType)
      .setLabelCol(featConfig.labelCol)
      .setFeaturesCol(featConfig.featuresCol)
      .setRandomForestNumericBoundaries(featConfig.numericBoundaries)
      .setRandomForestStringBoundaries(featConfig.stringBoundaries)
      .setScoringMetric(featConfig.scoringMetric)
      .setTrainPortion(featConfig.geneticConfig.trainPortion)
      .setKFold(featConfig.geneticConfig.kFold)
      .setSeed(featConfig.geneticConfig.seed)
      .setOptimizationStrategy(featConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(featConfig.geneticConfig.firstGenerationGenePool)
      .setNumberOfMutationGenerations(featConfig.geneticConfig.numberOfGenerations)
      .setNumberOfMutationsPerGeneration(featConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setNumberOfParentsToRetain(featConfig.geneticConfig.numberOfParentsToRetain)
      .setNumberOfMutationsPerGeneration(featConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setGeneticMixing(featConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(featConfig.geneticConfig.generationalMutationStrategy)
      .setMutationMagnitudeMode(featConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(featConfig.geneticConfig.fixedMutationValue)
      .evolveWithScoringDF()

    val bestModelData = modelResults.head
    val bestModelFeatureImportances = modelType match {
      case "classifier" => bestModelData.model.asInstanceOf[RandomForestClassificationModel].featureImportances.toArray
      case "regressor" => bestModelData.model.asInstanceOf[RandomForestRegressionModel].featureImportances.toArray
      case _ => throw new UnsupportedOperationException(
        s"The model type provided, '${featConfig.modelFamily}', is not supported.")
    }

    val importances = generateFrameReport(fields, bestModelFeatureImportances)

    (bestModelData, importances)

  }

}
