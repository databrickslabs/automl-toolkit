package com.databricks.labs.automl.ensemble.impl

import com.databricks.labs.automl.AutomationRunner
import com.databricks.labs.automl.ensemble.setting.StackingEnsembleSettings
import com.databricks.labs.automl.ensemble.{EnsembleLearner, EnsembleReturnType}
import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig}
import com.databricks.labs.automl.params.DataGeneration
import com.databricks.labs.automl.pipeline.PipelineInternalUtils._
import com.databricks.labs.automl.pipeline.{ColumnNameTransformer, SQLWrapperTransformer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer
private[ensemble] class StackingEnsembleLearner extends EnsembleLearner[StackingEnsembleSettings] {

  private lazy val PREDICTION_COL_NAME_SUFFIX = "_prediction"

  protected override def execute(stackingEnsembleSettings: StackingEnsembleSettings): Option[EnsembleReturnType] = {

    val familyRunnerOutput = FamilyRunner(
      stackingEnsembleSettings.inputData,
      stackingEnsembleSettings.weakLearnersConfigs)
      .executeWithPipeline()


//    val weakLearnerModelsWithPredictColumnsRenamed =  familyRunnerOutput
//        .bestPipelineModel
//        .map(item=> {
//            val predictionColNameTransformer = new ColumnNameTransformer()
//              .setInputColumns(Array("prediction"))
//              .setOutputColumns(Array(s"${item._1}$PREDICTION_COL_NAME_SUFFIX"))
//          val sqlTransformer = new SQLWrapperTransformer()
//            .setStatement(
//              s"""select ${stackingEnsembleSettings.inputData.columns.mkString(", ")}, ${item._1}$PREDICTION_COL_NAME_SUFFIX from __THIS__"""
//            )
//            mergePipelineModels(ArrayBuffer(item._2, predictionColNameTransformer, sqlTransformer))
//          })


    val metaModel = new Pipeline().setStages(Array())



    None
  }

//  private def buildMetaPipelineModel(featureCols: Array[String],
//                                     weakLearnersTransformedDf: DataFrame,
//                                     modelType: String): PipelineModel = {
//    val vaStage = new VectorAssembler()
//      .setInputCols(featureCols)
//      .setOutputCol("features")
//    val vectorizedDf = vaStage.transform(weakLearnersTransformedDf)
//
//    val config = ConfigurationGenerator.generateConfigFromMap("randomForest", "classifier", configurationOverrides)
//    val mainConfiguration = ConfigurationGenerator.generateMainConfig(config)
//    val runner = new AutomationRunner(vectorizedDf).setMainConfig(mainConfiguration)
//    val preppedDataOverride = DataGeneration(
//      vectorizedDf,
//      vectorizedDf.columns,
//      featureEngOutput.decidedModel
//    ).copy(modelType = x.predictionType)
//    runner.executeTuning(preppedDataOverride, isPipeline = true)
//
//  }
  override def validate(t: StackingEnsembleSettings): Unit = {

  }
}
