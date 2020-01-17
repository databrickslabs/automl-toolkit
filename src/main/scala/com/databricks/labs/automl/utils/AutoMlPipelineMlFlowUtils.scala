package com.databricks.labs.automl.utils

import java.nio.file.Paths

import com.databricks.labs.automl.params.{MLFlowConfig, MainConfig}
import com.databricks.labs.automl.pipeline.{PipelineStateCache, PipelineVars}
import com.databricks.labs.automl.tracking.MLFlowTracker
import org.apache.log4j.Logger
import org.apache.spark.ml.{PipelineModel, PredictionModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.mlflow.api.proto.Service

/**
  * @author Jas Bali
  * @since 0.6.1
  * Mlflow Utility for Pipeline tasks
  */
object AutoMlPipelineMlFlowUtils {

  @transient private val logger: Logger = Logger.getLogger(this.getClass)

  lazy final val AUTOML_INTERNAL_ID_COL = "automl_internal_id"

  case class ConfigByPipelineIdOutput(mainConfig: MainConfig,
                                      mlFlowRunId: String)

  def extractTopLevelColNames(schema: StructType): Array[String] =
    schema.fields.map(field => field.name)

  def getMainConfigByPipelineId(
    pipelineId: String
  ): ConfigByPipelineIdOutput = {
    val mainConfig = PipelineStateCache
      .getFromPipelineByIdAndKey(pipelineId, PipelineVars.MAIN_CONFIG.key)
      .asInstanceOf[MainConfig]
    if (mainConfig.mlFlowLoggingFlag) {
      val mlFlowRunId = PipelineStateCache
        .getFromPipelineByIdAndKey(pipelineId, PipelineVars.MLFLOW_RUN_ID.key)
        .asInstanceOf[String]
      ConfigByPipelineIdOutput(mainConfig, mlFlowRunId)
    } else {
      ConfigByPipelineIdOutput(mainConfig, null)
    }
  }

  def logTagsToMlFlow(pipelineId: String, tags: Map[String, String]): Unit = {
    val mlFlowRunIdAndConfig =
      AutoMlPipelineMlFlowUtils.getMainConfigByPipelineId(pipelineId)
    if (mlFlowRunIdAndConfig.mainConfig.mlFlowLoggingFlag) {
      val mlflowTracker = MLFlowTracker(
        mlFlowRunIdAndConfig.mainConfig.mlFlowConfig
      )
      val client = mlflowTracker.createHostedMlFlowClient()
      // Delete a tag first
      try {
        mlflowTracker
          .deleteCustomTags(
            client,
            mlFlowRunIdAndConfig.mlFlowRunId,
            tags.keys.toSet.toSeq
          )
      } catch {
        case ex: org.mlflow.tracking.MlflowHttpException => {
          logger.debug(s"MlFlow Tag deletion failed: ${ex.getBodyMessage}")
        }
      }
      //Create a new tag
      mlflowTracker
        .logCustomTags(client, mlFlowRunIdAndConfig.mlFlowRunId, tags)
    }
  }

  def getPipelinePathByRunId(runId: String,
                             mlFlowConfig: MLFlowConfig): String = {
    try {
      MLFlowTracker(mlFlowConfig)
        .createHostedMlFlowClient()
        .getRun(runId)
        .getData
        .getTagsList
        .toArray
        .map(item => item.asInstanceOf[Service.RunTag])
        .filter(
          item =>
            item.getKey
              .equals(PipelineMlFlowTagKeys.PIPELINE_MODEL_SAVE_PATH_KEY)
        )
        .head
        .getValue
    } catch {
      case e: Exception => {
        throw new RuntimeException(
          s"Exception in fetching Pipeline model path by MlFlow Run ID $runId",
          e
        )
      }
    }
  }

  def saveInferencePipelineDfAndLogToMlFlow(pipelineId: String,
                                            decidedModel: String,
                                            modelFamily: String,
                                            mlFlowModelSaveDirectory: String,
                                            finalPipelineModel: PipelineModel,
                                            originalDf: DataFrame): Unit = {
    val mlFlowRunIdAndConfig = getMainConfigByPipelineId(pipelineId: String)
    if (mlFlowRunIdAndConfig.mainConfig.mlFlowLoggingFlag) {
      // Log inference pipeline stages' names to MLFlow
      saveAllPipelineStagesToMlFlow(
        pipelineId,
        finalPipelineModel,
        mlFlowRunIdAndConfig.mainConfig
      )
      // Save Pipeline and log to MlFlow
      val modelDescriptor = s"$decidedModel" + "_" + s"$modelFamily"
      val baseDirectory = Paths.get(s"$mlFlowModelSaveDirectory/BestRun/")
      val pipelineDir =
        s"$baseDirectory${modelDescriptor}_${mlFlowRunIdAndConfig.mlFlowRunId}/BestPipeline/"
      val finalPipelineSavePath = Paths.get(pipelineDir).toString
      logger.info(
        s"Saving pipeline id $pipelineId to path $finalPipelineSavePath"
      )
      finalPipelineModel.save(finalPipelineSavePath)
      logger.info(
        s"Saved pipeline id $pipelineId to path $finalPipelineSavePath"
      )
      logTagsToMlFlow(
        pipelineId,
        Map(
          PipelineMlFlowTagKeys.PIPELINE_MODEL_SAVE_PATH_KEY -> finalPipelineSavePath
        )
      )
      // Save TrainingDf and log to MlFlow
      val trainDfBaseDirectory =
        Paths.get(s"$mlFlowModelSaveDirectory/FeatureEngineeredDataset/")
      val trainDfDir =
        s"$trainDfBaseDirectory${modelDescriptor}_${mlFlowRunIdAndConfig.mlFlowRunId}/data/"
      val finalFeatEngDfPath = Paths.get(trainDfDir).toString
      finalPipelineModel
        .transform(originalDf)
        .write
        .mode("overwrite")
        .format("delta")
        .save(finalFeatEngDfPath)
      logger.info(s"Saved feature engineered df to path $finalFeatEngDfPath")
      logTagsToMlFlow(
        pipelineId,
        Map(
          PipelineMlFlowTagKeys.PIPELINE_TRAIN_DF_PATH_KEY -> finalFeatEngDfPath
        )
      )
    }
  }

  private def saveAllPipelineStagesToMlFlow(pipelineId: String,
                                            finalPipelineModel: PipelineModel,
                                            mainConfig: MainConfig): Unit = {
    val finalPipelineStges =
      if (mainConfig.geneticConfig.trainSplitMethod == "kSample") {
        val ksamplerStagesPipelineHolder = "KSAMPLER_STAGER_PLACEHOLDER"
        val ksamplerPipelineStages = PipelineStateCache
          .getFromPipelineByIdAndKey(
            pipelineId,
            PipelineVars.KSAMPLER_STAGES.key
          )
          .asInstanceOf[String]
        // Interpolate to enter ksampler pipeline stages just before the modeling stage
        // to make sure pipeline stages are stringified in the order of their execution
        finalPipelineModel.stages
          .map(item => {
            if (item.isInstanceOf[PredictionModel[_, _]]) {
              ksamplerStagesPipelineHolder + ", \n" + item.getClass.getName
            } else {
              item.getClass.getName
            }
          })
          .mkString(", \n")
          .replace(ksamplerStagesPipelineHolder, ksamplerPipelineStages)
      } else {
        finalPipelineModel.stages.map(_.getClass.getName).mkString(", \n")
      }
    AutoMlPipelineMlFlowUtils
      .logTagsToMlFlow(
        pipelineId,
        Map(s"All_Stages_For_Pipeline_${pipelineId}" -> finalPipelineStges)
      )
  }

}
