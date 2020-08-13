package com.databricks.labs.automl.exploration.analysis.common.encoders

import org.apache.spark.ml.PipelineModel
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

import com.databricks.labs.automl.exploration.analysis.common.AnalysisUtilities
import com.databricks.labs.automl.exploration.analysis.common.structures.{
  NoParam,
  ParamWrapper
}

private[analysis] object ModelDebugStringParser {

  def modelDebugStringExtractor[T](model: T): String = {

    model match {
      case x: RandomForestClassificationModel => x.toDebugString
      case x: RandomForestRegressionModel     => x.toDebugString
      case x: DecisionTreeClassificationModel => x.toDebugString
      case x: DecisionTreeRegressionModel     => x.toDebugString
      case x: GBTClassificationModel          => x.toDebugString
      case x: GBTRegressionModel              => x.toDebugString
    }

  }

  def getModelDecisionText[T](
    model: T,
    vectorAssembler: ParamWrapper[VectorAssembler] = NoParam,
    vectorInputCols: ParamWrapper[Array[String]] = NoParam
  ): String = {

    // Get the debug string of the model
    val debugString = modelDebugStringExtractor(model)

    AnalysisUtilities
      .extractFieldsFromOptions(vectorAssembler, vectorInputCols)
      .zipWithIndex
      .map(x => s"feature ${x._2}" -> x._1)
      .foldLeft(debugString) {
        case (debugBody, (k, v)) => debugBody.replaceAll(k, v)
      }

  }

  def getModelDecisionTextFromPipeline(pipeline: PipelineModel): String = {

    val debugString = modelDebugStringExtractor(
      AnalysisUtilities.getModelFromPipeline(pipeline).last
    )

    val indexedFeatureNames =
      AnalysisUtilities
        .getFinalFeaturesFromPipeline(pipeline)
        .map { case (k, v) => (s"feature $v" -> k) }
    val initialReplace = indexedFeatureNames.foldLeft(debugString) {
      case (debugBody, (k, v)) => debugBody.replaceAll(k, v)
    }
    val stringIndexedValues =
      AnalysisUtilities.getStringIndexerMapping(pipeline)
    if (stringIndexedValues.isDefined) {
      stringIndexedValues.get.foldLeft(initialReplace) {
        case (debugStr, replacements) =>
          debugStr.replaceAll(replacements.after, replacements.before)
      }
    } else initialReplace
  }

}
