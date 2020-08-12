package com.databricks.labs.automl.ensemble.tuner
import com.databricks.labs.automl.ensemble.setting.StackingEnsembleSettings
import com.databricks.labs.automl.executor.config.InstanceConfig
import com.databricks.labs.automl.model.tools.structures.{TrainSplitReferences, TrainTestData}
import com.databricks.labs.automl.params.MainConfig
import com.databricks.labs.automl.pipeline.PipelineInternalUtils.{addTransformersToPipelineModels, mergePipelineModels}
import com.databricks.labs.automl.pipeline.{ColumnNameTransformer, DropColumnsTransformer, PipelineInternalUtils, SQLWrapperTransformer}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.VectorAssembler

class MetaLearnerFamilyRunner extends EnsembleFamilyRunnerLike
  [MetaLearnerFamilyRunnerReturnType, EnsembleFamilyRunnerLikeReturnType] {

  private lazy val PREDICTION_COL_NAME_SUFFIX = "_prediction"

  override def execute(stackingEnsembleSettings: StackingEnsembleSettings,
                       weakLearnersFamilyRunnerReturnType: Option[EnsembleFamilyRunnerLikeReturnType]): MetaLearnerFamilyRunnerReturnType = {

    val configs = Array(stackingEnsembleSettings.metaLearnerConfig)
    // convert configs into main configs
    val mainConfigs = getMainConfigs(configs)
    // Merge all weak learners into a single PipelineModel
    val weakLearnerPipelineModel: PipelineModel = mergePipelineModels(weakLearnersFamilyRunnerReturnType.get
      .familyFinalOutputWithPipeline
      .bestPipelineModel
      .map(item => {
        val predictionNewColName = s"${item._1}$PREDICTION_COL_NAME_SUFFIX"
        val predictionColNameTransformer = new ColumnNameTransformer()
          .setInputColumns(Array("prediction"))
          .setOutputColumns(Array(predictionNewColName))
        //Additional Check in case someone turns on verbose on FE and could result in collision
        // of intermediate FE columns (such as SI, OHE etc) between different models.
        val sqlTransformer = new SQLWrapperTransformer()
          .setStatement(
            s"""select ${stackingEnsembleSettings.inputData.columns.mkString(", ")}, $predictionNewColName from __THIS__"""
          )
        val thisConfig = getConfigByModelFamily(item._1, mainConfigs)
        val dropFeaturesTransformer = new DropColumnsTransformer()
            .setInputCols(Array(thisConfig.featuresCol))
            .setDebugEnabled(thisConfig.pipelineDebugFlag)
            .setPipelineId(thisConfig.pipelineId)
        addTransformersToPipelineModels(item._2, Array(predictionColNameTransformer, sqlTransformer, dropFeaturesTransformer))
      })
      .toArray)

    // Get Train Test Splits for meta learner
    val metaLearnerSplitsWithFeatures = weakLearnersFamilyRunnerReturnType
      .get
      .ensembleTunerSplits
      .get
      .getMetaLearnersSplits()
      .map(item => {
        TrainSplitReferences(
          item.kIndex,
          TrainTestData(
            weakLearnerPipelineModel.transform(item.data.train).select(mainConfigs(0).featuresCol, mainConfigs(0).labelCol),
            weakLearnerPipelineModel.transform(item.data.test).select(mainConfigs(0).featuresCol, mainConfigs(0).labelCol)
          ),
          item.paths
        )
      })


    // Generate PipelineModel to get features column for meta model
    val fePipelineModels = getFePipelineModels(stackingEnsembleSettings, mainConfigs)
    val feDfWithAllModelFeatures = weakLearnerPipelineModel
      .transform(stackingEnsembleSettings.inputData).select(mainConfigs(0).featuresCol, mainConfigs(0).labelCol)
    // Run Tuning for all weak learner models
    val (outputBuffer, pipelineConfigMap) = runTuningAndGetOutput(
      fePipelineModels,
      mainConfigs,
      configs,
      feDfWithAllModelFeatures,
      metaLearnerSplitsWithFeatures)

    MetaLearnerFamilyRunnerReturnType(
      withPipelineInferenceModel(
        stackingEnsembleSettings.inputData,
        unifyFamilyOutput(outputBuffer.toArray),
        configs,
        pipelineConfigMap.toMap
      ))
  }

  def getConfigByModelFamily(modelFamily: String,
       weakLearnersConfigs: Array[MainConfig]): MainConfig = {
      weakLearnersConfigs
      .filter(item => modelFamily.equals(item.modelFamily))
      .head
  }
}
