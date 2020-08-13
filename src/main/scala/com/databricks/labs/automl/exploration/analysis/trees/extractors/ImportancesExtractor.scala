package com.databricks.labs.automl.exploration.analysis.trees.extractors

import com.databricks.labs.automl.exploration.analysis.common.AnalysisUtilities
import com.databricks.labs.automl.exploration.analysis.common.structures.{
  FeatureImportanceData,
  NoParam,
  ParamWrapper
}
import org.apache.spark.ml.{PipelineModel, linalg}
import org.apache.spark.ml.classification.{
  DecisionTreeClassificationModel,
  GBTClassificationModel,
  RandomForestClassificationModel
}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{
  DecisionTreeRegressionModel,
  GBTRegressionModel,
  RandomForestRegressionModel
}

private[analysis] object ImportancesExtractor {

  import com.databricks.labs.automl.exploration.analysis.common.structures.PayloadType._

  private def castImportances[T](model: T): linalg.Vector = {

    model match {
      case x: DecisionTreeRegressionModel     => x.featureImportances
      case x: DecisionTreeClassificationModel => x.featureImportances
      case x: RandomForestRegressionModel     => x.featureImportances
      case x: RandomForestClassificationModel => x.featureImportances
      case x: GBTRegressionModel              => x.featureImportances
      case x: GBTClassificationModel          => x.featureImportances
    }

  }

  def extractImportancesFromModel[T](
    model: T,
    vectorAssembler: ParamWrapper[VectorAssembler] = NoParam,
    vectorInputCols: ParamWrapper[Array[String]] = NoParam
  ): FeatureImportanceData = {

    val importanceVector = castImportances(model)

    val vectorNames =
      AnalysisUtilities.extractFieldsFromOptions(
        vectorAssembler,
        vectorInputCols
      )

    FeatureImportanceData(importanceVector, vectorNames, MODEL, None)

  }

  def extractImportancesFromPipeline(
    pipeline: PipelineModel
  ): FeatureImportanceData = {

    val importanceVector = castImportances(
      AnalysisUtilities.getModelFromPipeline(pipeline).last
    )

    val vectorNames = AnalysisUtilities.getPipelineVectorFields(pipeline)

    val indexerMapping = AnalysisUtilities.getStringIndexerMapping(pipeline)

    FeatureImportanceData(
      importanceVector,
      vectorNames,
      PIPELINE,
      indexerMapping
    )

  }

}
