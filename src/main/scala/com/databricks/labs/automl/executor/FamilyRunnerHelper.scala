package com.databricks.labs.automl.executor

import com.databricks.labs.automl.exceptions.PipelineExecutionException
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig, InstanceConfigValidation}
import com.databricks.labs.automl.model.tools.split.PerformanceSettings
import com.databricks.labs.automl.params.{FamilyFinalOutput, FamilyFinalOutputWithPipeline, FamilyOutput, GenerationalReport, GenericModelReturn, GroupedModelReturn, MainConfig, TunerOutput}
import com.databricks.labs.automl.pipeline.{FeatureEngineeringOutput, FeatureEngineeringPipelineContext, PipelineMlFlowProgressReporter, PipelineStateCache, PipelineVars}
import com.databricks.labs.automl.tracking.{MLFlowReportStructure, MLFlowTracker}
import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, PipelineMlFlowTagKeys, SparkSessionWrapper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.lit

import scala.collection.mutable.ArrayBuffer

private[automl] trait FamilyRunnerHelper extends SparkSessionWrapper {

  /**
   * TODO All other tuner requirements should be called from here as well to enable fail fast
   * Enable fail fast for poorly configured environment
   * @param _parallelism Tuner Parallelism from Main Config
   * @throws java.lang.IllegalArgumentException if Invalid environment config
   */
  @throws(classOf[IllegalArgumentException])
  def validatePerformanceSettings(_parallelism: Int,
                                          modelFamily: String): Unit = {
    if(!SparkSession.builder().getOrCreate().sparkContext.isLocal) {
      if (modelFamily == "XGBoost") {
        PerformanceSettings.xgbWorkers(_parallelism)
      } else PerformanceSettings.optimalJVMModelPartitions(_parallelism)
    }
  }

  def addMainConfigToPipelineCache(mainConfig: MainConfig): Unit = {
    PipelineStateCache
      .addToPipelineCache(
        mainConfig.pipelineId,
        PipelineVars.MAIN_CONFIG.key,
        mainConfig
      )
  }

  def addMlFlowConfigForPipelineUse(mainConfig: MainConfig) = {
    addMainConfigToPipelineCache(mainConfig)
    if (mainConfig.mlFlowLoggingFlag) {
      val mlFlowRunId =
        MLFlowTracker(mainConfig).generateMlFlowRunId()
      PipelineStateCache
        .addToPipelineCache(
          mainConfig.pipelineId,
          PipelineVars.MLFLOW_RUN_ID.key,
          mlFlowRunId
        )
      AutoMlPipelineMlFlowUtils
        .logTagsToMlFlow(
          mainConfig.pipelineId,
          Map(
            s"${PipelineMlFlowTagKeys.PIPELINE_ID}"
              ->
              mainConfig.pipelineId
          )
        )
      PipelineMlFlowProgressReporter.starting(mainConfig.pipelineId)
    }
  }

  def getNewFamilyOutPut(output: TunerOutput,
                         instanceConfig: InstanceConfig): FamilyOutput = {
    new FamilyOutput(instanceConfig.modelFamily, output.mlFlowOutput) {
      override def modelReport: Array[GenericModelReturn] = output.modelReport

      override def generationReport: Array[GenerationalReport] =
        output.generationReport

      override def modelReportDataFrame: DataFrame =
        augmentDF(instanceConfig.modelFamily, output.modelReportDataFrame)

      override def generationReportDataFrame: DataFrame =
        augmentDF(instanceConfig.modelFamily, output.generationReportDataFrame)
    }
  }

  /**
   * Private method for adding a field to the output collection DataFrame to tell which model family generated
   * the data report.
   *
   * @param modelType The model type that was used for the experiment run
   * @param dataFrame the dataframe whose contents will be added to with a field of the literal model type that
   *                  generated the results.
   * @return a dataframe with the modeltype column added
   */
  private def augmentDF(modelType: String, dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("model", lit(modelType))
  }

  /**
   * Private method for unifying the outputs of each modeling run by family.  Allows for collapsing the array outputs
   * and unioning the DataFrames with additional information about what model was used to generate the summary report
   * data.
   *
   * @param outputArray output array of each modeling family's run
   * @return condensed report structure for all of the runs in a similar API return format.
   */
  def unifyFamilyOutput(outputArray: Array[FamilyOutput]
                               ): FamilyFinalOutput = {

    import spark.implicits._

    var modelReport = ArrayBuffer[GroupedModelReturn]()
    var generationReport = ArrayBuffer[GenerationalReport]()
    var modelReportDataFrame = spark.emptyDataset[ModelReportSchema].toDF
    var generationReportDataFrame =
      spark.emptyDataset[GenerationReportSchema].toDF
    var mlFlowOutput = ArrayBuffer[MLFlowReportStructure]()

    outputArray.map { x =>
      x.modelReport.map { y =>
        val model = y.model

        modelReport += GroupedModelReturn(
          modelFamily = x.modelType,
          hyperParams = y.hyperParams,
          model = model,
          score = y.score,
          metrics = y.metrics,
          generation = y.generation
        )
      }
      generationReport +: x.generationReport
      modelReportDataFrame.union(x.modelReportDataFrame)
      generationReportDataFrame.union(x.generationReportDataFrame)
      mlFlowOutput += x.mlFlowOutput
    }

    FamilyFinalOutput(
      modelReport = modelReport.toArray,
      generationReport = generationReport.toArray,
      modelReportDataFrame = modelReportDataFrame,
      generationReportDataFrame = generationReportDataFrame,
      mlFlowReport = mlFlowOutput.toArray
    )
  }

  def withPipelineInferenceModel(data: DataFrame,
                                 familyFinalOutput: FamilyFinalOutput,
                                  configs: Array[InstanceConfig],
                                  pipelineConfigs: Map[String, (FeatureEngineeringOutput, MainConfig)]
                                ): FamilyFinalOutputWithPipeline = {

    configs.foreach(InstanceConfigValidation(_).validate())

    val pipelineModels = scala.collection.mutable.Map[String, PipelineModel]()
    val bestMlFlowRunIds = scala.collection.mutable.Map[String, String]()
    configs.foreach(config => {
      val mainConfiguration = ConfigurationGenerator.generateMainConfig(config)
      val modelReport = familyFinalOutput.modelReport.filter(
        item => item.modelFamily.equals(config.modelFamily)
      )
      // Pipeline failure aware function
      withPipelineFailedException(mainConfiguration) {
        pipelineModels += config.modelFamily -> FeatureEngineeringPipelineContext
          .buildFullPredictPipeline(
            pipelineConfigs(config.modelFamily)._1,
            modelReport,
            pipelineConfigs(config.modelFamily)._2,
            data
          )
        if(mainConfiguration.mlFlowLoggingFlag) {
          bestMlFlowRunIds += config.modelFamily -> familyFinalOutput
            .mlFlowReport(0)
            .bestLog
            .runIdPayload(0)
            ._1
        }
      }
    })
    FamilyFinalOutputWithPipeline(
      familyFinalOutput,
      pipelineModels.toMap,
      bestMlFlowRunIds.toMap
    )
  }

  def withPipelineFailedException[T](mainConfig: MainConfig)(pipelineCode: => T): Unit = {
    try {
      pipelineCode
    } catch {
      case ex: Exception => {
        println(ex.printStackTrace())
        PipelineMlFlowProgressReporter.failed(
          mainConfig.pipelineId,
          ex.getMessage
        )
        throw PipelineExecutionException(mainConfig.pipelineId, ex)
      }
    }
  }

}
