package com.databricks.labs.automl.reports

import com.databricks.labs.automl.model.DecisionTreeTuner
import com.databricks.labs.automl.model.tools.split.{
  DataSplitCustodial,
  DataSplitUtility
}
import com.databricks.labs.automl.params.{MainConfig, TreeSplitReport}
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.sql.DataFrame

class DecisionTreeSplits(data: DataFrame,
                         featConfig: MainConfig,
                         modelType: String)
    extends ReportingTools {

  def runTreeSplitAnalysis(fields: Array[String]): TreeSplitReport = {

    val indexedFields = cleanupFieldArray(fields.zipWithIndex)

    val splitData = DataSplitUtility.split(
      data,
      featConfig.geneticConfig.kFold,
      featConfig.geneticConfig.trainSplitMethod,
      featConfig.labelCol,
      featConfig.geneticConfig.deltaCacheBackingDirectory,
      featConfig.geneticConfig.splitCachingStrategy,
      featConfig.modelFamily
    )

    val (modelResults, modelStats) = new DecisionTreeTuner(
      data,
      splitData,
      modelType
    ).setLabelCol(featConfig.labelCol)
      .setFeaturesCol(featConfig.featuresCol)
      .setTreesNumericBoundaries(featConfig.numericBoundaries)
      .setTreesStringBoundaries(featConfig.stringBoundaries)
      .setScoringMetric(featConfig.scoringMetric)
      .setTrainPortion(featConfig.geneticConfig.trainPortion)
      .setKFold(featConfig.geneticConfig.kFold)
      .setSeed(featConfig.geneticConfig.seed)
      .setOptimizationStrategy(featConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(
        featConfig.geneticConfig.firstGenerationGenePool
      )
      .setNumberOfMutationGenerations(
        featConfig.geneticConfig.numberOfGenerations
      )
      .setNumberOfMutationsPerGeneration(
        featConfig.geneticConfig.numberOfMutationsPerGeneration
      )
      .setNumberOfParentsToRetain(
        featConfig.geneticConfig.numberOfParentsToRetain
      )
      .setNumberOfMutationsPerGeneration(
        featConfig.geneticConfig.numberOfMutationsPerGeneration
      )
      .setGeneticMixing(featConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(
        featConfig.geneticConfig.generationalMutationStrategy
      )
      .setMutationMagnitudeMode(featConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(featConfig.geneticConfig.fixedMutationValue)
      .evolveWithScoringDF()

    val bestModelData = modelResults.head

    val treeModelBest = modelType match {
      case "regressor" =>
        bestModelData.model.asInstanceOf[DecisionTreeRegressionModel]
      case "classifier" =>
        bestModelData.model.asInstanceOf[DecisionTreeClassificationModel]
      case _ =>
        throw new UnsupportedOperationException(
          s"modelType $modelType is not supported for DecisionTrees."
        )
    }

    val treeModelString = modelType match {
      case "regressor" =>
        bestModelData.model
          .asInstanceOf[DecisionTreeRegressionModel]
          .toDebugString
      case "classifier" =>
        bestModelData.model
          .asInstanceOf[DecisionTreeClassificationModel]
          .toDebugString
      case _ =>
        throw new UnsupportedOperationException(
          s"modelType $modelType is not supported for DecisionTrees."
        )
    }

    val featureImportances = modelType match {
      case "regressor" =>
        bestModelData.model
          .asInstanceOf[DecisionTreeRegressionModel]
          .featureImportances
          .toArray
      case "classifier" =>
        bestModelData.model
          .asInstanceOf[DecisionTreeClassificationModel]
          .featureImportances
          .toArray
      case _ =>
        throw new UnsupportedOperationException(
          s"modelType $modelType is not supported for DecisionTrees."
        )
    }

    val importances = generateFrameReport(fields, featureImportances)

    val mappedModelString =
      generateDecisionTextReport(treeModelString, indexedFields)

    DataSplitCustodial.cleanCachedInstances(splitData, featConfig)

    TreeSplitReport(mappedModelString, importances, treeModelBest)

  }

}
