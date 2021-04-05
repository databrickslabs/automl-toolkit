package com.databricks.labs.automl.executor

import com.databricks.labs.automl.AutomationRunner
import com.databricks.labs.automl.exceptions.PipelineExecutionException
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig, InstanceConfigValidation}
import com.databricks.labs.automl.model.tools.ModelUtils
import com.databricks.labs.automl.model.tools.split.PerformanceSettings
import com.databricks.labs.automl.params._
import com.databricks.labs.automl.pipeline._
import com.databricks.labs.automl.tracking.{MLFlowReportStructure, MLFlowTracker}
import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, PipelineMlFlowTagKeys, SparkSessionWrapper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, SparkSession}
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
    extends FamilyRunnerHelper {

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

    configs.foreach(InstanceConfigValidation(_) validate())

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

  /**
    *
    * @return grouped results same as execute [[FamilyFinalOutputWithPipeline]] but
    *         also contains a map of model family and best pipeline model (along with mlflow Run ID)
    *         based on optimization strategy settings
    */
  def executeWithPipeline(): FamilyFinalOutputWithPipeline = {

    configs.foreach { x =>
      InstanceConfigValidation(x).validate()
    }

    val outputBuffer = ArrayBuffer[FamilyOutput]()

    val pipelineConfigMap = scala.collection.mutable
      .Map[String, (FeatureEngineeringOutput, MainConfig)]()
    configs.foreach { x =>
      val mainConfiguration = ConfigurationGenerator.generateMainConfig(x)
      validatePerformanceSettings(
        mainConfiguration.geneticConfig.parallelism,
        mainConfiguration.modelFamily
      )
      val runner = new AutomationRunner(data).setMainConfig(mainConfiguration)

      // Perform cardinality check if the model type is a tree-based family and update the
      // numeric mappings to handle the maxBins issue for nominal and categorical data.

      x.modelFamily.toLowerCase.replaceAll("\\s", "") match {
        case "randomforest" | "trees" | "gbt" | "xgboost" => {
          val updatedNumBoundaries = ModelUtils.resetTreeBinsSearchSpace(
            data,
            x.algorithmConfig.numericBoundaries,
            x.genericConfig.fieldsToIgnoreInVector,
            x.genericConfig.labelCol,
            x.genericConfig.featuresCol
          )
          runner.setNumericBoundaries(updatedNumBoundaries)
        }
        case _ => Unit
      }

      // Setup MLflow Run
      addMlFlowConfigForPipelineUse(mainConfiguration)
      // Pipeline failure aware function
      withPipelineFailedException(mainConfiguration) {
        //Get feature engineering pipeline and transform it to get feature engineered dataset
        val featureEngOutput = FeatureEngineeringPipelineContext
          .generatePipelineModel(data, mainConfiguration)
        val featureEngineeredDf = featureEngOutput.transformedForTrainingDf
        val preppedDataOverride = DataGeneration(
          featureEngineeredDf,
          featureEngineeredDf.columns,
          featureEngOutput.decidedModel
        ).copy(modelType = x.predictionType)

        val output = runner.executeTuning(preppedDataOverride, isPipeline = true)

        outputBuffer += getNewFamilyOutPut(output, x)
        pipelineConfigMap += x.modelFamily -> (featureEngOutput, mainConfiguration)
      }
    }
    withPipelineInferenceModel(
      data,
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
  def generateFeatureEngineeredPipeline(
    verbose: Boolean = false
  ): Map[String, PipelineModel] = {

    configs.foreach { x =>
      InstanceConfigValidation(x).validate()
    }

    val featureEngineeredMap =
      scala.collection.mutable.Map[String, PipelineModel]()
    configs.foreach { x =>
      val mainConfiguration = ConfigurationGenerator.generateMainConfig(x)
      addMainConfigToPipelineCache(mainConfiguration)
      val featureEngOutput =
        FeatureEngineeringPipelineContext.generatePipelineModel(
          data,
          mainConfiguration,
          verbose,
          isFeatureEngineeringOnly = true
        )
      val finalPipelineModel =
        FeatureEngineeringPipelineContext.addUserReturnViewStage(
          featureEngOutput.pipelineModel,
          mainConfiguration,
          featureEngOutput.pipelineModel.transform(data),
          featureEngOutput.originalDfViewName
        )
      featureEngineeredMap += x.modelFamily -> finalPipelineModel
    }
    featureEngineeredMap.toMap
  }
}

/**
  * Companion Object allowing for class instantiation throfugh configs either as an Instance config or Map overrides
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
