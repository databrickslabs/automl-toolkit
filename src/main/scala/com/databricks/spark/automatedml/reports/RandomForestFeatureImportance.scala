package com.databricks.spark.automatedml.reports

import com.databricks.spark.automatedml.model.RandomForestTuner
import com.databricks.spark.automatedml.params.{MainConfig, RandomForestModelsWithResults}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class RandomForestFeatureImportance(data: DataFrame, fieldListing: Array[String], featConfig: MainConfig)
  extends RandomForestTuner(data, featConfig.modelType) {


  private def generateFrameReport(columns: Array[String], importances: Array[Double]): DataFrame = {
    import spark.sqlContext.implicits._
    sc.parallelize(columns zip importances).toDF("Columns", "Importances").orderBy($"Importances".desc)
  }

  def runFeatureImportances(): (RandomForestModelsWithResults, DataFrame) = {

    val (modelResults, modelStats) = new RandomForestTuner(data, featConfig.modelType)
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

    val bestModelData = modelResults(0)
    val bestModelFeatureImportances = featConfig.modelType match {
      case "classifier" => bestModelData.model.asInstanceOf[RandomForestClassificationModel].featureImportances.toArray
      case "regressor" => bestModelData.model.asInstanceOf[RandomForestRegressionModel].featureImportances.toArray
      case _ => throw new UnsupportedOperationException(s"The model type provided, '${featConfig.modelType}', is not supported.")
    }

    val importances = generateFrameReport(fieldListing, bestModelFeatureImportances)
      .withColumn("Importances", col("Importances") * 100)
      .withColumn("Columns", split(col("Columns"), "_si$")(0))

    (modelResults(0), importances)

  }

}
