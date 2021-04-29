package com.databricks.labs.automl.ensemble.impl

import com.databricks.labs.automl.ensemble.exception.EnsembleValidationExceptions
import com.databricks.labs.automl.ensemble.setting.StackingEnsembleSettings
import com.databricks.labs.automl.ensemble.tuner.{MetaLearnerFamilyRunner, WeakLearnersFamilyRunner}
import com.databricks.labs.automl.ensemble.{EnsembleLearner, EnsembleReturnType}
import com.databricks.labs.automl.executor.config.InstanceConfigValidation
import com.databricks.labs.automl.pipeline.PipelineInternalUtils
import org.apache.spark.ml.Pipeline

private[ensemble] class StackingEnsembleLearner extends EnsembleLearner[StackingEnsembleSettings] {

  protected override def execute(stackingEnsembleSettings: StackingEnsembleSettings): Option[EnsembleReturnType] = {

    val weakLearnersReturn = WeakLearnersFamilyRunner().execute(stackingEnsembleSettings)

    val metaLearnerFamilyRunner = MetaLearnerFamilyRunner().execute(stackingEnsembleSettings, Some(weakLearnersReturn))

//    TODO: write ensemble model to MLflow
    Some(EnsembleReturnType(
      PipelineInternalUtils
        .mergePipelineModels(
          Array(
            metaLearnerFamilyRunner.weakLeanerPipelineModel,
            metaLearnerFamilyRunner.familyFinalOutputWithPipeline.bestPipelineModel.head._2)),
          null,
          metaLearnerFamilyRunner.familyFinalOutputWithPipeline.familyFinalOutput,
          weakLearnersReturn.familyFinalOutputWithPipeline.familyFinalOutput))
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
    // Validate weak learners
    stackingEnsembleSettings.weakLearnersConfigs.foreach{InstanceConfigValidation(_).validate()}
    //Validate meta learner
    InstanceConfigValidation(stackingEnsembleSettings.weakLearnersConfigs.head).validate()

    val trainPortions = stackingEnsembleSettings
      .weakLearnersConfigs
      .map(item => (item.tunerConfig.tunerTrainPortion, item.tunerConfig.tunerTrainSplitMethod))
      .toSet


    // Ensure all weak learners and meta learner have sample split configs
    if (trainPortions.size > 1)
      throw EnsembleValidationExceptions.TRAIN_PORTION_EXCEPTION

    // Ensure Ksample is not set
    trainPortions
      .head
      ._2 match {
      case "kSample" => throw EnsembleValidationExceptions.KSAMPLE_NOT_SUPPORTED
      case _ =>
    }
  }
}
