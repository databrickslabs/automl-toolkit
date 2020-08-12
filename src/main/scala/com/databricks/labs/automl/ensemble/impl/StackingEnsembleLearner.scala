package com.databricks.labs.automl.ensemble.impl

import com.databricks.labs.automl.ensemble.exception.EnsembleValidationExceptions
import com.databricks.labs.automl.ensemble.setting.StackingEnsembleSettings
import com.databricks.labs.automl.ensemble.tuner.{EnsembleTunerSplits, WeakLearnersFamilyRunner}
import com.databricks.labs.automl.ensemble.{EnsembleLearner, EnsembleReturnType}
import com.databricks.labs.automl.executor.config.InstanceConfigValidation
import com.databricks.labs.automl.model.tools.structures.{TrainSplitReferences, TrainTestData}
import com.databricks.labs.automl.pipeline.PipelineInternalUtils._
import com.databricks.labs.automl.pipeline.{ColumnNameTransformer, PipelineInternalUtils, SQLWrapperTransformer}
import org.apache.spark.ml.{Pipeline, PipelineModel}

import scala.collection.mutable.ArrayBuffer

private[ensemble] class StackingEnsembleLearner extends EnsembleLearner[StackingEnsembleSettings] {



  protected override def execute(stackingEnsembleSettings: StackingEnsembleSettings): Option[EnsembleReturnType] = {

    val weakLearnersReturn = WeakLearnersFamilyRunner().execute(stackingEnsembleSettings)










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

  override def validate(stackingEnsembleSettings: StackingEnsembleSettings): Unit = {
//    TODO: weak and meta
//    configs.foreach{InstanceConfigValidation(_).validate()}
    val trainPortions = stackingEnsembleSettings
      .weakLearnersConfigs
      .map(_.tunerConfig.tunerTrainPortion)
      .toSet
      .size

   val trainSplitMethod = stackingEnsembleSettings
    .weakLearnersConfigs
    .map(_.tunerConfig.tunerTrainSplitMethod)
    .toSet
    .size


    if (trainPortions > 1 || trainSplitMethod > 1)
      throw EnsembleValidationExceptions.TRAIN_PORTION_EXCEPTION

  }
}
