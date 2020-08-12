package com.databricks.labs.automl.exploration.analysis.trees.extractors

import com.databricks.labs.automl.exploration.analysis.common.AnalysisUtilities
import com.databricks.labs.automl.exploration.analysis.common.structures.{
  NoParam,
  ParamWrapper,
  VisualizationOutput
}
import com.databricks.labs.automl.exploration.analysis.trees.scripts.HTMLGenerators
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

private[analysis] object VisualizationExtractor {

  def extractModelVisualizationDataFromModel[T](
    model: T,
    mode: String,
    vectorAssembler: ParamWrapper[VectorAssembler] = NoParam,
    vectorInputCols: ParamWrapper[Array[String]] = NoParam
  ): Array[VisualizationOutput] = {

    require(
      vectorAssembler.asOption.isDefined || vectorInputCols.asOption.isDefined,
      s"Either the VectorAssembler used to build the model to test must be supplied or an Array of field" +
        s"names from the Vector must be supplied."
    )

    val fieldNames = vectorAssembler.asOption match {
      case Some(vectorAssembler: VectorAssembler) =>
        vectorAssembler.getInputCols
      case _ => vectorInputCols.asOption.get
    }

    val treeVisualization = model match {
      case x: DecisionTreeRegressionModel =>
        new TreeExtractor(x.asInstanceOf[DecisionTreeRegressionModel])
          .getVisualizationData(fieldNames)
      case x: DecisionTreeClassificationModel =>
        new TreeExtractor(x.asInstanceOf[DecisionTreeClassificationModel])
          .getVisualizationData(fieldNames)
      case x: RandomForestRegressionModel =>
        new TreeExtractor(x.asInstanceOf[RandomForestRegressionModel])
          .getVisualizationData(fieldNames)
      case x: RandomForestClassificationModel =>
        new TreeExtractor(x.asInstanceOf[RandomForestClassificationModel])
          .getVisualizationData(fieldNames)
      case x: GBTRegressionModel =>
        new TreeExtractor(x.asInstanceOf[GBTRegressionModel])
          .getVisualizationData(fieldNames)
      case x: GBTClassificationModel =>
        new TreeExtractor(x.asInstanceOf[GBTClassificationModel])
          .getVisualizationData(fieldNames)
      case _ =>
        throw new UnsupportedOperationException(
          "The model supplied is not supported."
        )
    }

    treeVisualization.zipWithIndex.map(
      x =>
        VisualizationOutput(
          x._2,
          HTMLGenerators.createD3TreeVisualization(
            x._1,
            mode,
            ModelConfigExtractor.extractModelData(model)
          )
      )
    )

  }

  def extractModelVisualizationDataFromPipeline(
    pipe: PipelineModel,
    mode: String
  ): Array[VisualizationOutput] = {

    val model = AnalysisUtilities.getModelFromPipeline(pipe).last

    new PipelineExtractor(pipe).getVisualizationData.zipWithIndex
      .map(
        x =>
          VisualizationOutput(
            x._2,
            HTMLGenerators.createD3TreeVisualization(
              x._1,
              mode,
              ModelConfigExtractor.extractModelData(model)
            )
        )
      )

  }

}
