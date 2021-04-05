package com.databricks.labs.automl.ensemble.tuner
import com.databricks.labs.automl.ensemble.setting.StackingEnsembleSettings
import com.databricks.labs.automl.model.tools.structures.{TrainSplitReferences, TrainTestData}
import com.databricks.labs.automl.params.MainConfig
import com.databricks.labs.automl.pipeline.PipelineInternalUtils.{addTransformersToPipelineModels, createPipelineModelFromStages}
import com.databricks.labs.automl.pipeline.{AutoMlOutputDatasetTransformer, ColumnNameTransformer, DropColumnsTransformer, SQLWrapperTransformer}
import org.apache.spark.ml.{PipelineModel, PredictionModel, Predictor, Transformer}

class MetaLearnerFamilyRunner extends EnsembleFamilyRunnerLike
  [MetaLearnerFamilyRunnerReturnType, EnsembleFamilyRunnerLikeReturnType] {

  private lazy val PREDICTION_COL_NAME_SUFFIX = "_prediction"

  override def execute(stackingEnsembleSettings: StackingEnsembleSettings,
                       weakLearnersFamilyRunnerReturnType: Option[EnsembleFamilyRunnerLikeReturnType]):
  MetaLearnerFamilyRunnerReturnType = {

    val configs = Array(stackingEnsembleSettings.metaLearnerConfig.get)
    // convert configs into main configs
    val mainConfigs = getMainConfigs(configs)
    // Merge all weak learners into a single PipelineModel
    val allWeakLearnerStages =
      weakLearnersFamilyRunnerReturnType
        .get
        .familyFinalOutputWithPipeline
        .bestPipelineModel
        .flatMap(_._2.stages)
        .toArray
    val lastOutputDfColumnStage =
      allWeakLearnerStages
        .reverse
        .find(_.uid.startsWith("AutoMlOutputDatasetTransformer"))
        .get
    val weakLearnerStagesWithoutLastOutputFormat = allWeakLearnerStages.filterNot(_.uid.equals(lastOutputDfColumnStage.uid))

    val predictionColsRenameStages = weakLearnersFamilyRunnerReturnType.get
      .familyFinalOutputWithPipeline
      .bestPipelineModel
      .flatMap(item => {
        val predictionNewColName = s"${item._1}$PREDICTION_COL_NAME_SUFFIX"
        val predictionColNameTransformer = new ColumnNameTransformer()
          .setInputColumns(Array("prediction"))
          .setOutputColumns(Array(predictionNewColName))
        //Additional Check in case someone turns on verbose on FE and could result in collision
        // of intermediate FE columns (such as SI, OHE etc) between different models.
//        val sqlTransformer = new SQLWrapperTransformer()
//          .setStatement(
//            s"""select ${stackingEnsembleSettings.inputData.columns.mkString(", ")},
//               |$predictionNewColName from __THIS__""".stripMargin
//          )
        val thisConfig = getConfigByModelFamily(item._1, mainConfigs)
        val dropFeaturesTransformer = new DropColumnsTransformer()
            .setInputCols(Array(thisConfig.featuresCol))
            .setDebugEnabled(thisConfig.pipelineDebugFlag)
            .setPipelineId(thisConfig.pipelineId)
        Array(predictionColNameTransformer,
//            sqlTransformer,
            dropFeaturesTransformer)
      }).toArray

     val finalWeakLearnerPipelineModel = createPipelineModelFromStages(
       weakLearnerStagesWithoutLastOutputFormat ++
       predictionColsRenameStages ++
       Array(lastOutputDfColumnStage))

    val weakLearnerPipelineWithoutFe = createPipelineModelFromStages(
      finalWeakLearnerPipelineModel
        .stages
        .filter(_.isInstanceOf[PredictionModel[_, _]])
        ++
        predictionColsRenameStages)

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
            weakLearnerPipelineWithoutFe
              .transform(item.data.train)
              .select(mainConfigs(0).labelCol, getAllMetaLearnerFeatureCols(weakLearnersFamilyRunnerReturnType):_*),
            weakLearnerPipelineWithoutFe
              .transform(item.data.test)
              .select(mainConfigs(0).labelCol, getAllMetaLearnerFeatureCols(weakLearnersFamilyRunnerReturnType):_*)
          ),
          item.paths
        )
      })
    val feDfWithAllModelFeatures = finalWeakLearnerPipelineModel
      .transform(stackingEnsembleSettings.inputData)
      .select(mainConfigs(0).labelCol, getAllMetaLearnerFeatureCols(weakLearnersFamilyRunnerReturnType):_*)
    // Generate PipelineModel to get features column for meta model
    val fePipelineModels = getFePipelineModels(feDfWithAllModelFeatures, mainConfigs)
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
        pipelineConfigMap.toMap),
      finalWeakLearnerPipelineModel
    )
  }

  def getConfigByModelFamily(modelFamily: String,
       weakLearnersConfigs: Array[MainConfig]): MainConfig = {
      weakLearnersConfigs
      .filter(item => modelFamily.equals(item.modelFamily))
      .head
  }

  private def getAllMetaLearnerFeatureCols(weakLearnersFamilyRunnerReturnType:
                                           Option[EnsembleFamilyRunnerLikeReturnType]): Array[String] = {
    weakLearnersFamilyRunnerReturnType
      .get
      .familyFinalOutputWithPipeline
      .bestPipelineModel
      .map(item => s"${item._1}$PREDICTION_COL_NAME_SUFFIX")
      .toArray
  }
}

private[ensemble] object MetaLearnerFamilyRunner {
  def apply(): MetaLearnerFamilyRunner = new MetaLearnerFamilyRunner()
}