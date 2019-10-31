package com.databricks.labs.automl.executor

import com.databricks.labs.automl.AutomationRunner
import com.databricks.labs.automl.exceptions.PipelineExecutionException
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig}
import com.databricks.labs.automl.params._
import com.databricks.labs.automl.pipeline._
import com.databricks.labs.automl.tracking.{MLFlowReportStructure, MLFlowTracker}
import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, PipelineMlFlowTagKeys, SparkSessionWrapper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer

case class ModelReportSchema(generation: Int, score: Double, model: String)

case class GenerationReportSchema(model_family: String,
                                  model_type: String,
                                  generation: Int,
                                  generation_mean_score: Double,
                                  generation_std_dev_score: Double,
                                  model: String)

/**
  * @constructor Determine the best possible SparkML model for an ML task by supplying a DataFrame and an Array of
  *              InstanceConfig objects that have been defined with ConfigurationGenerator()
  * @author Ben Wilson, Databricks
  * @since 0.5.0.3
  * @param data A Spark DataFrame that contains feature columns and a label column
  * @param configs The configuration for each of the model types that are to be tested, stored in an Array.
  * @example
  * ```
  * val data: DataFrame = spark.table("db.test")
  * val mapOverrides: Map[String, Any] = Map("labelCol" -> "finalLabelCol", "tunerNumberOfMutationsPerGeneration" -> 5)
  *
  * val randomForestConfig = ConfigurationGenerator.generateConfigFromMap("RandomForest", "classifier", mapOverrides)
  * val logRegConfig = ConfigurationGenerator.generateConfigFromMap("LogisticRegression", "classifier", mapOverrides)
  * val treeConfig = ConfigurationGenerator.generateConfigFromMap("Trees", "classifier", mapOverrides)
  *
  * val runner = new FamilyRunner(data, Array(randomForestConfig, logRegConfig, treeConfig)).execute
  * ```
  */
class FamilyRunner(data: DataFrame, configs: Array[InstanceConfig])
    extends SparkSessionWrapper {

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
  private def unifyFamilyOutput(outputArray: Array[FamilyOutput]): FamilyFinalOutput = {

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
    *
    * @deprecated Use [[executeWithPipeline()]] instead.
    *             Start using executeWithPipeline to leverage Pipeline semantics
    *
    *             Main method for executing the family runs as configured.
    * @return FamilyOutput object that reports the results of each of the family modeling runs.
    */
  @Deprecated
  def execute(): FamilyFinalOutput = {

    val outputBuffer = ArrayBuffer[FamilyOutput]()

    configs.foreach { x =>
      val mainConfiguration = ConfigurationGenerator.generateMainConfig(x)

      val runner = new AutomationRunner(data)
        .setMainConfig(mainConfiguration)

      val preppedData = runner.prepData()

      val preppedDataOverride = preppedData.copy(modelType = x.predictionType)

      val output = runner.executeTuning(preppedDataOverride)

      outputBuffer += getNewFamilyOutPut(output, x)
    }
    unifyFamilyOutput(outputBuffer.toArray)

  }

  private def addMainConfigToPipelineCache(mainConfig: MainConfig): Unit = {
    PipelineStateCache
      .addToPipelineCache(
        mainConfig.pipelineId,
        PipelineVars.MAIN_CONFIG.key, mainConfig)
  }


  private def addMlFlowConfigForPipelineUse(mainConfig: MainConfig) = {
    addMainConfigToPipelineCache(mainConfig)
    if(mainConfig.mlFlowLoggingFlag) {
      val mlFlowRunId = MLFlowTracker(mainConfig.mlFlowConfig).generateMlFlowRunId()
      PipelineStateCache
        .addToPipelineCache(
          mainConfig.pipelineId,
          PipelineVars.MLFLOW_RUN_ID.key, mlFlowRunId)
      AutoMlPipelineMlFlowUtils
        .logTagsToMlFlow(
          mainConfig.pipelineId,
          Map(s"${PipelineMlFlowTagKeys.PIPELINE_ID}"
            ->
            mainConfig.pipelineId
          ))
      PipelineMlFlowProgressReporter.starting(mainConfig.pipelineId)
    }
  }

  /**
    *
    * @return grouped results same as execute [[FamilyFinalOutputWithPipeline]] but
    *         also contains a map of model family and best pipeline model (along with mlflow Run ID)
    *         based on optimization strategy settings
    */
  def executeWithPipeline(): FamilyFinalOutputWithPipeline = {

    val outputBuffer = ArrayBuffer[FamilyOutput]()

    val pipelineConfigMap = scala.collection.mutable.Map[String, (FeatureEngineeringOutput, MainConfig)]()
    configs.foreach { x =>
      val mainConfiguration = ConfigurationGenerator.generateMainConfig(x)
      val runner = new AutomationRunner(data).setMainConfig(mainConfiguration)
      // Setup MLflow Run
      addMlFlowConfigForPipelineUse(mainConfiguration)
      try {
        //Get feature engineering pipeline and transform it to get feature engineered dataset
        val featureEngOutput = FeatureEngineeringPipelineContext.generatePipelineModel(data, mainConfiguration)
        val featureEngineeredDf = featureEngOutput.transformedForTrainingDf

        val preppedDataOverride = DataGeneration(
          featureEngineeredDf,
          featureEngineeredDf.columns,
          featureEngOutput.decidedModel
        ).copy(modelType = x.predictionType)

        val output = runner.executeTuning(preppedDataOverride, isPipeline = true)

        outputBuffer += getNewFamilyOutPut(output, x)
        pipelineConfigMap += x.modelFamily -> (featureEngOutput, mainConfiguration)
      } catch {
        case ex: Exception => {
          PipelineMlFlowProgressReporter.failed(mainConfiguration.pipelineId, ex.getMessage)
          throw PipelineExecutionException(mainConfiguration.pipelineId, ex)
        }
      }
    }
    withPipelineInferenceModel(
      unifyFamilyOutput(outputBuffer.toArray),
      configs,
      pipelineConfigMap.toMap
    )
  }

  /**
    * @param verbose: If set to true, any dataset transformed with this feature engineered pipeline will include all
    *               input columns for the vector assembler stage.
    * @return Generates feature engineering pipeline for a given configuration under a given Model Family
    *         Note: It does not trigger any Model training.
    */
  def generateFeatureEngineeredPipeline(verbose: Boolean = false): Map[String, PipelineModel] = {
    val featureEngineeredMap = scala.collection.mutable.Map[String, PipelineModel]()
    configs.foreach { x =>
      val mainConfiguration = ConfigurationGenerator.generateMainConfig(x)
      addMainConfigToPipelineCache(mainConfiguration)
      val featureEngOutput = FeatureEngineeringPipelineContext.generatePipelineModel(data, mainConfiguration, verbose, isFeatureEngineeringOnly = true)
      val finalPipelineModel = FeatureEngineeringPipelineContext.addUserReturnViewStage(
        featureEngOutput.pipelineModel,
        mainConfiguration,
        featureEngOutput.pipelineModel.transform(data),
        featureEngOutput.originalDfViewName)
      featureEngineeredMap += x.modelFamily -> finalPipelineModel
    }
    featureEngineeredMap.toMap
  }

  def withPipelineInferenceModel(familyFinalOutput: FamilyFinalOutput,
                                 configs: Array[InstanceConfig],
                                 pipelineConfigs: Map[String, (FeatureEngineeringOutput, MainConfig)]):

  FamilyFinalOutputWithPipeline = {
    val pipelineModels = scala.collection.mutable.Map[String, PipelineModel]()
    val bestMlFlowRunIds = scala.collection.mutable.Map[String, String]()
    configs.foreach(config => {
      val modelReport = familyFinalOutput.modelReport.filter(item => item.modelFamily.equals(config.modelFamily))
      pipelineModels += config.modelFamily -> FeatureEngineeringPipelineContext
        .buildFullPredictPipeline(
          pipelineConfigs(config.modelFamily)._1,
          modelReport,
          pipelineConfigs(config.modelFamily)._2,
          data)
      bestMlFlowRunIds += config.modelFamily -> familyFinalOutput.mlFlowReport(0).bestLog.runIdPayload(0)._1
    })
    FamilyFinalOutputWithPipeline(familyFinalOutput, pipelineModels.toMap, bestMlFlowRunIds.toMap)
  }
}

/**
  * Companion Object allowing for class instantiation through configs either as an Instance config or Map overrides
  * collection.
  */
object FamilyRunner {

  def apply(data: DataFrame, configs: Array[InstanceConfig]): FamilyRunner =
    new FamilyRunner(data, configs)

  def apply(data: DataFrame,
            modelFamily: String,
            predictionType: String,
            configs: Array[Map[String, Any]]): FamilyRunner = {

    val configBuffer = ArrayBuffer[InstanceConfig]()

    configs.foreach { x =>
      configBuffer += ConfigurationGenerator.generateConfigFromMap(
        modelFamily,
        predictionType,
        x
      )
    }

    new FamilyRunner(data, configBuffer.toArray)
  }

}
