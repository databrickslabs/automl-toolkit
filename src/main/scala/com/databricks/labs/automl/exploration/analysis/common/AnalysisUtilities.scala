package com.databricks.labs.automl.exploration.analysis.common

import com.databricks.labs.automl.exploration.analysis.common.structures.StringIndexerMappings
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{
  DecisionTreeClassificationModel,
  GBTClassificationModel,
  LogisticRegressionModel,
  RandomForestClassificationModel
}
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.regression.{
  DecisionTreeRegressionModel,
  GBTRegressionModel,
  LinearRegressionModel,
  RandomForestRegressionModel
}
import com.databricks.labs.automl.exploration.analysis.common.structures.{
  NoParam,
  ParamWrapper
}

import scala.collection.mutable.ArrayBuffer

private[analysis] object AnalysisUtilities {
  private def parseStringIndexers(
    pipeline: PipelineModel,
    stringIndexerStageValues: ArrayBuffer[StringIndexerMappings] =
      ArrayBuffer.empty[StringIndexerMappings]
  ): ArrayBuffer[StringIndexerMappings] = {

    pipeline.stages
      .collect {
        case x: StringIndexerModel =>
          val indexer = x.asInstanceOf[StringIndexerModel]
          stringIndexerStageValues += StringIndexerMappings(
            indexer.getInputCol,
            indexer.getOutputCol
          )
        case x: PipelineModel =>
          parseStringIndexers(x, stringIndexerStageValues)
      }
    stringIndexerStageValues

  }

  def getStringIndexerMapping(
    pipeline: PipelineModel
  ): Option[Array[StringIndexerMappings]] = {

    Option(parseStringIndexers(pipeline).result.toArray)

  }

  def getModelFromPipeline(pipeline: PipelineModel): Array[Any] = {

    pipeline.stages.collect {
      case x: RandomForestClassificationModel =>
        x.asInstanceOf[RandomForestClassificationModel]
      case x: RandomForestRegressionModel =>
        x.asInstanceOf[RandomForestRegressionModel]
      case x: DecisionTreeClassificationModel =>
        x.asInstanceOf[DecisionTreeClassificationModel]
      case x: DecisionTreeRegressionModel =>
        x.asInstanceOf[DecisionTreeRegressionModel]
      case x: GBTRegressionModel      => x.asInstanceOf[GBTRegressionModel]
      case x: GBTClassificationModel  => x.asInstanceOf[GBTClassificationModel]
      case x: LogisticRegressionModel => x.asInstanceOf[LogisticRegressionModel]
      case x: LinearRegressionModel   => x.asInstanceOf[LinearRegressionModel]
      case x: PipelineModel           => getModelFromPipeline(x)
    }
  }

  def getPipelineVectorFields(pipeline: PipelineModel): Array[String] = {
    pipeline.stages.collect {
      case x: VectorAssembler => x.asInstanceOf[VectorAssembler].getInputCols
      case x: PipelineModel   => getPipelineVectorFields(x)
    }.flatten
  }

  def getFinalFeaturesFromPipeline(
    pipeline: PipelineModel
  ): Map[String, Int] = {

    pipeline.stages
      .collect {
        case x: VectorAssembler =>
          x.asInstanceOf[VectorAssembler].getInputCols.zipWithIndex.toMap
        case x: PipelineModel => getFinalFeaturesFromPipeline(x)

      }
      .flatten
      .toMap

  }

  def extractFieldsFromOptions(
    vectorAssembler: ParamWrapper[VectorAssembler] = NoParam,
    vectorInputCols: ParamWrapper[Array[String]] = NoParam
  ): Array[String] = {

    require(
      vectorAssembler.asOption.isDefined || vectorInputCols.asOption.isDefined,
      s"Either a VectorAssembler " +
        s"instance or an Array of field names that were used to build the Vector is required."
    )

    if (vectorAssembler.asOption.isDefined) {
      vectorAssembler.asOption.get.getInputCols
    } else { vectorInputCols.asOption.get }

  }

}
