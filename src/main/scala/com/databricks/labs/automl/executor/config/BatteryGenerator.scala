package com.databricks.labs.automl.executor.config

import com.databricks.labs.automl.executor.config.ModelSelector.ModelSelector
import com.databricks.labs.automl.executor.config.PredictionType.PredictionType
import com.databricks.labs.automl.params.MainConfig

import scala.collection.mutable.ArrayBuffer

/**
  * A class and companion object to generate configurations for a particular modeling type.  Developer API.
  * Currently available modeling types: `regressor` and `classifier`
  *
  * @example
  *          ```
  *          val classifierConfigs = new BatterryGenerator("regressor").setModelsToTest(Array("RandomForest", "GBT"))
  *                                     .generateDefaultConfigs(
  *                                                             "labelColumn",
  *                                                             "features",
  *                                                             "https://instance.host.com",
  *                                                             "regressorTest",
  *                                                             "<mytoken>",
  *                                                             "/model/save/dir",
  *                                                             "/inference/save/dir",
  *                                                             "full",
  *                                                             "_best_model",
  *                                                             true,
  *                                                             false)
  *          ```
  * @author Ben Wilson, Databricks
  * @since 0.5.0.3
  * @param predictionType modeling type to generate: either 'regressor' or 'classifier'
  */
class BatteryGenerator(predictionType: String)
    extends BatteryDefaults
    with ConfigurationDefaults {

  private[config] val batteryType: PredictionType =
    BatteryGenerator.predictionTypeAssignment(predictionType)

  // Instantiate the Defaults
  private var _modelsToTest: Array[ModelSelector] = modelSelection(batteryType)

  // Override setter for defining models to be run
  def setModelsToTest(modelFamilies: Array[String]): this.type = {

    // Define the allowable collection of model types
    val allowableModels = modelSelection(batteryType)

    // Create a holding collection
    var modelCollection = ArrayBuffer[ModelSelector]()

    // Run validation on supplied models to create instance types
    modelFamilies foreach { x =>
      val definedModel = modelTypeEvaluator(x, predictionType)
      require(
        allowableModels.contains(definedModel),
        s"ModelsToTest setting $x is not allowable.  Must be one of:" +
          s"${allowableModels.flatMap(x => x.toString).mkString(", ")}"
      )
      modelCollection += definedModel
    }

    // Overwrite the defined defaults for restricting the families of models based on user input.
    _modelsToTest = modelCollection.toArray
    this
  }

  def setModelsToTest(modelFamilies: String*): this.type = {
    setModelsToTest(modelFamilies.toArray)
    this
  }

  def getModelsToTest: Array[ModelSelector] = _modelsToTest

  /**
    * Public method that requires definition of settings that will need to be unique for the run, but using
    * all other default settings for modeling and algorithm tuning (quick-start)
    *
    * @param labelCol Label column in the DataFrame
    * @param featuresCol Name of the to-be-generated feature vector column
    * @param mlFlowTrackingURI Tracking host address
    * @param mlFlowExperimentName Unique name for the experiment to be run
    * @param mlFlowAPIToken API token for the MLFlow service
    * @param modelSaveDirectory Blob storage location for saving build models
    * @param inferenceConfigSaveLocation Blob storage location for saving the Inference Config to reproduce the run
    * @param mlFlowLoggingMode modes supported: "tuningOnly", "bestOnly", "full"
    * @param mlFlowBestSuffix string to append to a new experiment run that captures only the best run found in its
    *                         own experiment location
    * @param mlFlowLoggingFlag setting on whether or not to log anything to mlflow
    * @param mlFlowLogArtifactsFlag setting on whether or not to log the model artifacts to mlflow
    * @param fieldsToIgnoreInModeling optional field that can be populated by an array of column names to be ignored
    *                                 for the purposes of modeling, but retained in the final result dataframe(s)
    * @return An array of configuration objects with (mostly) default settings for each model type that has been
    *         supplied in the configuration of this class.
    */
  def generateDefaultConfigs(labelCol: String,
                             featuresCol: String,
                             mlFlowTrackingURI: String,
                             mlFlowExperimentName: String,
                             mlFlowAPIToken: String,
                             modelSaveDirectory: String,
                             inferenceConfigSaveLocation: String,
                             mlFlowLoggingMode: String = "full",
                             mlFlowBestSuffix: String = "_best",
                             mlFlowLoggingFlag: Boolean = true,
                             mlFlowLogArtifactsFlag: Boolean = true,
                             fieldsToIgnoreInModeling: Array[String] =
                               Array.empty[String]): Array[InstanceConfig] = {

    val defaultBuffer: ArrayBuffer[InstanceConfig] =
      ArrayBuffer[InstanceConfig]()

    _modelsToTest foreach { x =>
      val defaultGenericConfig: GenericConfig =
        new GenericConfigGenerator(predictionType)
          .setLabelCol(labelCol)
          .setFeaturesCol(featuresCol)
          .setFieldsToIgnoreInVector(fieldsToIgnoreInModeling)
          .getConfig

      val modelSpecificConfig = new ConfigurationGenerator(
        modelToStandardizedString(x),
        predictionType,
        defaultGenericConfig
      ).setMlFlowTrackingURI(mlFlowTrackingURI)
        .setMlFlowAPIToken(mlFlowAPIToken)
        .setMlFlowExperimentName(mlFlowExperimentName)
        .setMlFlowModelSaveDirectory(modelSaveDirectory)
        .setMlFlowLoggingFlag(mlFlowLoggingFlag)
        .setMlFlowLogArtifactsFlag(mlFlowLogArtifactsFlag)

      defaultBuffer += modelSpecificConfig.getInstanceConfig
    }

    defaultBuffer.toArray

  }

  /**
    * Public method for generating main config objects based on a family
    *
    * @param labelCol Label column in the DataFrame
    * @param featuresCol Name of the to-be-generated feature vector column
    * @param mlFlowTrackingURI Tracking host address
    * @param mlFlowExperimentName Unique name for the experiment to be run
    * @param mlFlowAPIToken API token for the MLFlow service
    * @param modelSaveDirectory Blob storage location for saving build models
    * @param inferenceConfigSaveLocation Blob storage location for saving the Inference Config to reproduce the run
    * @param mlFlowLoggingMode modes supported: "tuningOnly", "bestOnly", "full"
    * @param mlFlowBestSuffix string to append to a new experiment run that captures only the best run found in its
    *                         own experiment location
    * @param mlFlowLoggingFlag setting on whether or not to log anything to mlflow
    * @param mlFlowLogArtifactsFlag setting on whether or not to log the model artifacts to mlflow
    * @param fieldsToIgnoreInModeling optional field that can be populated by an array of column names to be ignored
    *                                 for the purposes of modeling, but retained in the final result dataframe(s)
    * @return Array of MainConfig objects
    */
  @deprecated(
    "Main Config accessor will be replaced in future versions by InstanceConfig()."
  )
  def generateDefaultMainConfigs(labelCol: String,
                                 featuresCol: String,
                                 mlFlowTrackingURI: String,
                                 mlFlowExperimentName: String,
                                 mlFlowAPIToken: String,
                                 modelSaveDirectory: String,
                                 inferenceConfigSaveLocation: String,
                                 mlFlowLoggingMode: String = "full",
                                 mlFlowBestSuffix: String = "_best",
                                 mlFlowLoggingFlag: Boolean = true,
                                 mlFlowLogArtifactsFlag: Boolean = true,
                                 fieldsToIgnoreInModeling: Array[String] =
                                   Array.empty[String]): Array[MainConfig] = {

    val mainConfigBuffer: ArrayBuffer[MainConfig] = ArrayBuffer[MainConfig]()

    val instanceConfigs = generateDefaultConfigs(
      labelCol,
      featuresCol,
      mlFlowTrackingURI,
      mlFlowExperimentName,
      mlFlowAPIToken,
      modelSaveDirectory,
      inferenceConfigSaveLocation,
      mlFlowLoggingMode,
      mlFlowBestSuffix,
      mlFlowLoggingFlag,
      mlFlowLogArtifactsFlag,
      fieldsToIgnoreInModeling
    )

    instanceConfigs foreach { x =>
      mainConfigBuffer += ConfigurationGenerator.generateMainConfig(x)
    }
    mainConfigBuffer.toArray

  }

}

object BatteryGenerator extends ConfigurationDefaults {

  def apply(predictionType: String): BatteryGenerator =
    new BatteryGenerator(predictionType)

  def predictionTypeAssignment(predictionType: String): PredictionType = {
    predictionTypeEvaluator(predictionType)
  }

}
