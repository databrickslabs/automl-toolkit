package com.databricks.labs.automl.tracking

import java.io.File
import java.nio.file.Paths

import com.databricks.labs.automl.inference.InferenceConfig._
import com.databricks.labs.automl.inference.{InferenceModelConfig, InferenceTools}
import com.databricks.labs.automl.params.{GenericModelReturn, MLFlowConfig, MainConfig}
import com.databricks.labs.automl.utils.PipelineMlFlowTagKeys
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostRegressionModel}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, GBTRegressionModel, LinearRegressionModel, RandomForestRegressionModel}
import org.mlflow.api.proto.Service
import org.mlflow.api.proto.Service.CreateRun
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.creds._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

class MLFlowTracker extends InferenceTools {

  private var _mlFlowTrackingURI: String = _
  private var _mlFlowExperimentName: String = "default"
  private var _mlFlowHostedAPIToken: String = _
  private var _modelSaveDirectory: String = _
  private var _logArtifacts: Boolean = false
  private var _mlFlowLoggingMode: String = _
  private var _mlFlowBestSuffix: String = _
  private var _mlFlowCustomRunTags: Map[String, String] = Map.empty

  final private val HOSTED_NAMESPACE = List("databricks.com", "databricks.net")

  def setMlFlowTrackingURI(value: String): this.type = {
    _mlFlowTrackingURI = value
    this
  }

  def setMlFlowHostedAPIToken(value: String): this.type = {
    _mlFlowHostedAPIToken = value
    this
  }

  def setMlFlowExperimentName(value: String): this.type = {
    _mlFlowExperimentName = value
    this
  }

  def setModelSaveDirectory(value: String): this.type = {
    _modelSaveDirectory = value
    this
  }

  def logArtifactsOn(): this.type = {
    _logArtifacts = true
    this
  }

  def logArtifactsOff(): this.type = {
    _logArtifacts = false
    this
  }

  def setMlFlowLoggingMode(value: String): this.type = {
    _mlFlowLoggingMode = value
    this
  }

  def setMlFlowBestSuffix(value: String): this.type = {
    _mlFlowBestSuffix = value
    this
  }

  def setMlFlowCustomRunTags(value: Map[String, String]): this.type = {
    _mlFlowCustomRunTags = value
    this
  }

  //Intentionally not providing a getter for an API token.

  def getMlFlowTrackingURI: String = _mlFlowTrackingURI
  def getMlFlowExperimentName: String = _mlFlowExperimentName
  def getModelSaveDirectory: String = _modelSaveDirectory
  def getArtifactLogSetting: Boolean = _logArtifacts
  def getMlFlowLoggingMode: String = _mlFlowLoggingMode
  def getMlFlowBestSuffix: String = _mlFlowBestSuffix
  def getMlFlowCustomRunTags: Map[String, String] = _mlFlowCustomRunTags

  /**
    * Method for either getting an existing experiment by name, or creating a new one by name and returning the id
    *
    * @param client: MlflowClient to get access to the mlflow service agent
    * @return the experiment id from either an existing run or the newly created one.
    */
  private def getOrCreateExperimentId(client: MlflowClient,
                                      experimentName: String =
                                        _mlFlowExperimentName): String = {

    val experiment = client.getExperimentByName(experimentName)
    if (experiment.isPresent) experiment.get().getExperimentId
    else client.createExperiment(experimentName)

  }

  def createHostedMlFlowClient(): MlflowClient = {

    val hosted: Boolean = HOSTED_NAMESPACE.exists(_mlFlowTrackingURI.contains)

    if (hosted) {
      new MlflowClient(
        new BasicMlflowHostCreds(_mlFlowTrackingURI, _mlFlowHostedAPIToken)
      )
    } else {
      new MlflowClient(_mlFlowTrackingURI)
    }
  }

  private def generateMlFlowRun(client: MlflowClient,
                                experimentID: String,
                                runIdentifier: String,
                                runName: String,
                                sourceVer: String): String = {
    val request: CreateRun.Builder = CreateRun
      .newBuilder()
      .setExperimentId(experimentID)
      .setStartTime(System.currentTimeMillis())
      .addTags(Service.RunTag.newBuilder().setKey("mlflow.runName").setValue(runName).build())
      .addTags(Service.RunTag.newBuilder().setKey("mlflow.source.name").setValue(runIdentifier).build())
      .addTags(Service.RunTag.newBuilder().setKey("mlflow.source.version").setValue(sourceVer).build())
    val run = client.createRun(request.build())
    run.getRunId
  }

  def generateMlFlowRunId(): String = {
    val client = createHostedMlFlowClient()
    val experimentId = getOrCreateExperimentId(client, _mlFlowExperimentName + _mlFlowBestSuffix).toString
    client.createRun(experimentId).getRunId
  }

  private def createFusePath(path: String): String = {
    path.replace("dbfs:", "/dbfs")
  }

   def logCustomTags(client: MlflowClient,
                            runId: String,
                            tags: Map[String, String]): Unit = {
    if (tags.nonEmpty) {
      tags.foreach { case (k, v) => client.setTag(runId, k, v) }
    }
   }

  def deleteCustomTags(client: MlflowClient,
                    runId: String,
                    tagKeys: Seq[String]): Unit = {
    if (tagKeys.nonEmpty) {
      tagKeys.foreach(k => client.deleteTag(runId, k))
    }
  }

  /**
    * Private method for saving an individual model, creating a Fuse mount for it, and registering the artifact.
    *
    * @param client MlFlow client that has been registered.
    * @param path Path in blob store for saving the SparkML Model
    * @param runId Unique runID for the run to log model artifacts to.
    * @param modelReturn Modeling payload for the run in order to extract the specific model type
    * @param modelDescriptor Text Assignment for the model family + type of model that was run
    * @param modelId Unique uuid identifier for the model.
    */
  private def saveModel(client: MlflowClient,
                        path: String,
                        runId: String,
                        modelReturn: GenericModelReturn,
                        modelDescriptor: String,
                        modelId: String): Unit = {

    println(s"Model will be saved to path $path")
    modelDescriptor match {
      case "regressor_RandomForest" =>
        modelReturn.model
          .asInstanceOf[RandomForestRegressionModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case "classifier_RandomForest" =>
        modelReturn.model
          .asInstanceOf[RandomForestClassificationModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case "regressor_XGBoost" =>
        modelReturn.model
          .asInstanceOf[XGBoostRegressionModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case "classifier_XGBoost" =>
        modelReturn.model
          .asInstanceOf[XGBoostClassificationModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case "regressor_GBT" =>
        modelReturn.model
          .asInstanceOf[GBTRegressionModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case "classifier_GBT" =>
        modelReturn.model
          .asInstanceOf[GBTClassificationModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case "classifier_MLPC" =>
        modelReturn.model
          .asInstanceOf[MultilayerPerceptronClassificationModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case "regressor_LinearRegression" =>
        modelReturn.model
          .asInstanceOf[LinearRegressionModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case "classifier_LogisticRegression" =>
        modelReturn.model
          .asInstanceOf[LogisticRegressionModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case "regressor_SVM" =>
        modelReturn.model
          .asInstanceOf[LinearSVCModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case "regressor_Trees" =>
        modelReturn.model
          .asInstanceOf[DecisionTreeRegressionModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case "classifier_Trees" =>
        modelReturn.model
          .asInstanceOf[DecisionTreeClassificationModel]
          .write
          .overwrite()
          .save(path)
        if (_logArtifacts)
          client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, "ModelSaveLocation", path)
        client.setTag(runId, "TrainingPayload", modelReturn.toString)
      case _ =>
        throw new UnsupportedOperationException(
          s"Model Type $modelDescriptor is not supported for mlflow logging."
        )
    }
  }

  /**
    * Public method for logging a model, parameters, and metrics to MlFlow
    *
    * @param runData Full collection parameters, results, and models for the autoML experiment
    * @param modelFamily Type of Model Family used (e.g. "RandomForest")
    * @param modelType Type of Model used (e.g. "regression")
    */
  def logMlFlowDataAndModels(
    runData: Array[GenericModelReturn],
    modelFamily: String,
    modelType: String,
    inferenceSaveLocation: String,
    optimizationStrategy: String
  ): MLFlowReportStructure = {

    val dummyLog =
      MLFlowReturn(createHostedMlFlowClient(), "none", Array(("none", 0.0)))

    val bestLog = _mlFlowLoggingMode match {
      case "tuningOnly" =>
        dummyLog
      case _ =>
        logBest(
          runData,
          modelFamily,
          modelType,
          inferenceSaveLocation,
          optimizationStrategy
        )
    }

    val fullLog = _mlFlowLoggingMode match {
      case "bestOnly" => dummyLog
      case _ =>
        logTuning(runData, modelFamily, modelType, inferenceSaveLocation)
    }

    MLFlowReportStructure(fullLog = fullLog, bestLog = bestLog)

  }

  /**
    * This method does not save any artifacts or inference configs.
    * For the Best Model logging mode, it logs params and metrics to a given mlFlowRunId
    * For the tuning logging mode, it logs params and metrics to separate mlFlowRunIds
    * @param mlFlowRunId
    * @param runData
    * @param modelFamily
    * @param modelType
    * @param optimizationStrategy
    * @return
    */
  def logMlFlowForPipeline(mlFlowRunId: String,
                           runData: Array[GenericModelReturn],
                           modelFamily: String,
                           modelType: String,
                           optimizationStrategy: String
                          ): MLFlowReportStructure = {
    val dummyLog =
      MLFlowReturn(createHostedMlFlowClient(), "none", Array(("none", 0.0)))

    val bestLog = _mlFlowLoggingMode match {
      case "tuningOnly" =>
        dummyLog
      case _ =>
        logBestForPipeline(
          mlFlowRunId,
          runData,
          modelFamily,
          modelType,
          optimizationStrategy
        )
    }

    val fullLog = _mlFlowLoggingMode match {
      case "bestOnly" => dummyLog
      case _ =>
        logTuningForPipeline(runData, modelFamily, modelType)
    }
    MLFlowReportStructure(fullLog = fullLog, bestLog = bestLog)
  }

  private def logBestForPipeline(
                      mlFlowRunId: String,
                      runData: Array[GenericModelReturn],
                      modelFamily: String,
                      modelType: String,
                      optimizationStrategy: String): MLFlowReturn = {
    val mlflowLoggingClient = createHostedMlFlowClient()

    val experimentId = getOrCreateExperimentId(
      mlflowLoggingClient,
      _mlFlowExperimentName + _mlFlowBestSuffix
    ).toString

    val bestModel = getBestModel(optimizationStrategy, runData)

    val runIdPayload = Array((mlFlowRunId, bestModel.score))

    val modelHyperParams = bestModel.hyperParams.keys
    val metrics = bestModel.metrics.keys

    modelHyperParams.foreach { x =>
      val valueData = bestModel.hyperParams(x)
      mlflowLoggingClient.logParam(mlFlowRunId, x, valueData.toString)
    }
    metrics.foreach { x =>
      val valueData = bestModel.metrics(x)
      mlflowLoggingClient.logMetric(mlFlowRunId, x, valueData.toString.toDouble)
    }

    val modelDescriptor = s"${modelType}_$modelFamily"
    mlflowLoggingClient.logParam(mlFlowRunId, "modelType", modelDescriptor)

    mlflowLoggingClient.logParam(mlFlowRunId, "generation", "Best")

    // Log custom tags if present
    if (_mlFlowCustomRunTags.nonEmpty) {
      logCustomTags(mlflowLoggingClient, mlFlowRunId, _mlFlowCustomRunTags)
    }

    MLFlowReturn(
      mlflowLoggingClient,
      experimentId,
      runIdPayload)
  }

  private def getBestModel(optimizationStrategy: String,
                           runData: Array[GenericModelReturn]): GenericModelReturn = {
    optimizationStrategy match {
      case "minimize" => runData.sortWith(_.score < _.score)(0)
      case _          => runData.sortWith(_.score > _.score)(0)
    }
  }

  private def logBest(runData: Array[GenericModelReturn],
                      modelFamily: String,
                      modelType: String,
                      inferenceSaveLocation: String,
                      optimizationStrategy: String): MLFlowReturn = {

    val bestModel = getBestModel(optimizationStrategy, runData)
    val mlflowLoggingClient = createHostedMlFlowClient()
    val experimentId = getOrCreateExperimentId(
      mlflowLoggingClient,
      _mlFlowExperimentName + _mlFlowBestSuffix
    ).toString

    var totalVersion =
      mlflowLoggingClient.getExperiment(experimentId).getRunsCount

    val baseDirectory = Paths.get(s"${_modelSaveDirectory}/BestRun/").toString

    val modelDescriptor = s"${modelType}_$modelFamily"

    //TODO(Jas): This needs to be synchronized to make sure two a true runVersion is generated
    val runVersion: Int = totalVersion + 1

    val runId = generateMlFlowRun(
      mlflowLoggingClient,
      experimentId,
      modelDescriptor,
      "BestRun",
      runVersion.toString
    )

    val runIdPayload = Array((runId, bestModel.score))

    val modelHyperParams = bestModel.hyperParams.keys
    val metrics = bestModel.metrics.keys

    modelHyperParams.foreach { x =>
      val valueData = bestModel.hyperParams(x)
      mlflowLoggingClient.logParam(runId, x, valueData.toString)
    }
    metrics.foreach { x =>
      val valueData = bestModel.metrics(x)
      mlflowLoggingClient.logMetric(runId, x, valueData.toString.toDouble)
    }

    mlflowLoggingClient.logParam(runId, "modelType", modelDescriptor)

    val modelDir = s"$baseDirectory${modelDescriptor}_$runId/bestModel"

    saveModel(
      mlflowLoggingClient,
      modelDir,
      runId,
      bestModel,
      modelDescriptor,
      "BestRun"
    )
    mlflowLoggingClient.logParam(runId, "generation", "Best")

    // Log custom tags if present
    if (_mlFlowCustomRunTags.nonEmpty) {
      logCustomTags(mlflowLoggingClient, runId, _mlFlowCustomRunTags)
    }

    //Inference data save
    val inferencePath = Paths
      .get(s"$inferenceSaveLocation/$experimentId/${_mlFlowBestSuffix}/")
      .toString
    val inferenceLocation = inferencePath + runId + _mlFlowBestSuffix
    val inferenceMlFlowConfig = getInternalMlFlowConfig(baseDirectory)
    val inferenceModelConfig = getInferenceModelConfig(
      modelFamily,
      modelType,
      "mlflow",
      inferenceMlFlowConfig,
      runId,
      modelDir)
    setInferenceModelConfig(inferenceModelConfig)
    setInferenceConfigStorageLocation(inferenceLocation)

    val inferenceConfig = getInferenceConfig

    val inferenceConfigAsJSON = convertInferenceConfigToJson(inferenceConfig)

    val inferenceConfigAsDF = convertInferenceConfigToDataFrame(inferenceConfig)

    //Save the inference config to the save location
    println(s"Inference DF will be saved to $inferenceLocation")
    inferenceConfigAsDF.write.save(inferenceLocation)

    mlflowLoggingClient.setTag(
      runId,
      "InferenceConfig",
      inferenceConfigAsJSON.compactJson
    )

    mlflowLoggingClient.setTag(
      runId,
      "InferenceDataFrameLocation",
      inferenceLocation
    )

    MLFlowReturn(mlflowLoggingClient, experimentId, runIdPayload)

  }

  private def getInferenceModelConfig(modelFamily: String,
                                      modelType: String,
                                      modelLoadMethod: String,
                                      inferenceMlFlowConfig: MLFlowConfig,
                                      mlFlowRunId: String,
                                      modelPathLocation: String): InferenceModelConfig = {
    InferenceModelConfig(
      modelFamily = modelFamily,
      modelType = modelType,
      modelLoadMethod = "mlflow",
      mlFlowConfig = inferenceMlFlowConfig,
      mlFlowRunId = mlFlowRunId,
      modelPathLocation = modelPathLocation
    )
  }
  private def getInternalMlFlowConfig(baseDirectory: String): MLFlowConfig = {
    MLFlowConfig(
      mlFlowTrackingURI = _mlFlowTrackingURI,
      mlFlowExperimentName = _mlFlowExperimentName,
      mlFlowAPIToken = _mlFlowHostedAPIToken,
      mlFlowModelSaveDirectory = baseDirectory,
      mlFlowLoggingMode = _mlFlowLoggingMode,
      mlFlowBestSuffix = _mlFlowBestSuffix,
      mlFlowCustomRunTags = _mlFlowCustomRunTags
    )
  }

  private def logTuningForPipeline(
                        runData: Array[GenericModelReturn],
                        modelFamily: String,
                        modelType: String): MLFlowReturn = {

    val runIdPayloadBuffer = ArrayBuffer[(String, Double)]()

    val mlflowLoggingClient = createHostedMlFlowClient()
    val experimentId = getOrCreateExperimentId(mlflowLoggingClient).toString

    var totalVersion =
      mlflowLoggingClient.getExperiment(experimentId).getRunsCount

    val generationSet = mutable.Set[Int]()
    runData.map(x => generationSet += x.generation)
    val uniqueGenerations = generationSet.result.toArray.sortWith(_ < _)

    val modelDescriptor = s"${modelType}_$modelFamily"

    // loop through each generation and log the data
    uniqueGenerations.foreach { g =>
      // get the runs from this generation
      val currentGen = runData.filter(x => x.generation == g)
      var withinRunId = 0
      // Execute these writes in parallel.
      val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(10))
      val generations = currentGen.par
      generations.tasksupport = taskSupport
      generations.foreach { x =>
        totalVersion += 1
        val uniqueRunIdent =
          s"${modelFamily}_${modelType}_${x.generation.toString}_${withinRunId.toString}_${x.score.toString}"
        val runName = "run_" + x.generation.toString + "_" + withinRunId.toString
        val runId = generateMlFlowRun(
          mlflowLoggingClient,
          experimentId,
          uniqueRunIdent,
          runName,
          totalVersion.toString
        )
        runIdPayloadBuffer += Tuple2(runId, x.score)
        val hyperParamKeys = x.hyperParams.keys
        hyperParamKeys.foreach { k =>
          val valueData = modelFamily match {
            case "MLPC" =>
              x.hyperParams(k) match {
                case "layers" =>
                  x.hyperParams(k).asInstanceOf[Array[Int]].mkString(",")
                case _ => x.hyperParams(k)
              }
            case _ => x.hyperParams(k)
          }
          //val valueData = x.hyperParams(k)
          mlflowLoggingClient.logParam(runId, k, valueData.toString)
        }
        val metricKeys = x.metrics.keys
        metricKeys.foreach { k =>
          val valueData = x.metrics(k)
          mlflowLoggingClient.logMetric(runId, k, valueData.toString.toDouble)
        }
        mlflowLoggingClient.logParam(runId, "modelType", modelDescriptor)
        // log the generation
        mlflowLoggingClient.logParam(runId, "generation", x.generation.toString)
        // Log custom tags if present
        if (_mlFlowCustomRunTags.nonEmpty) {
          logCustomTags(mlflowLoggingClient, runId, _mlFlowCustomRunTags)
        }
        withinRunId += 1
      }
    }
    MLFlowReturn(
      mlflowLoggingClient,
      experimentId,
      runIdPayloadBuffer.result().toArray
    )
  }

  private def logTuning(runData: Array[GenericModelReturn],
                        modelFamily: String,
                        modelType: String,
                        inferenceSaveLocation: String): MLFlowReturn = {

    val runIdPayloadBuffer = ArrayBuffer[(String, Double)]()

    val mlflowLoggingClient = createHostedMlFlowClient()

    val experimentId = getOrCreateExperimentId(mlflowLoggingClient).toString

    var totalVersion =
      mlflowLoggingClient.getExperiment(experimentId).getRunsCount

    val generationSet = mutable.Set[Int]()
    runData.map(x => generationSet += x.generation)
    val uniqueGenerations = generationSet.result.toArray.sortWith(_ < _)

    val baseDirectory = Paths.get(s"${_modelSaveDirectory}/").toString

    val modelDescriptor = s"${modelType}_$modelFamily"

    // loop through each generation and log the data
    uniqueGenerations.foreach { g =>
      // get the runs from this generation
      val currentGen = runData.filter(x => x.generation == g)

      var withinRunId = 0

      // Execute these writes in parallel.
      val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(10))
      val generations = currentGen.par
      generations.tasksupport = taskSupport

      generations.foreach { x =>
        totalVersion += 1

        val uniqueRunIdent =
          s"${modelFamily}_${modelType}_${x.generation.toString}_${withinRunId.toString}_${x.score.toString}"

        val runName = "run_" + x.generation.toString + "_" + withinRunId.toString

        val runId = generateMlFlowRun(
          mlflowLoggingClient,
          experimentId,
          uniqueRunIdent,
          runName,
          totalVersion.toString
        )

        runIdPayloadBuffer += Tuple2(runId, x.score)

        val hyperParamKeys = x.hyperParams.keys

        hyperParamKeys.foreach { k =>
          val valueData = modelFamily match {
            case "MLPC" =>
              x.hyperParams(k) match {
                case "layers" =>
                  x.hyperParams(k).asInstanceOf[Array[Int]].mkString(",")
                case _ => x.hyperParams(k)
              }
            case _ => x.hyperParams(k)
          }

          //val valueData = x.hyperParams(k)
          mlflowLoggingClient.logParam(runId, k, valueData.toString)
        }
        val metricKeys = x.metrics.keys

        metricKeys.foreach { k =>
          val valueData = x.metrics(k)
          mlflowLoggingClient.logMetric(runId, k, valueData.toString.toDouble)
        }

        mlflowLoggingClient.logParam(runId, "modelType", modelDescriptor)

        // Generate a new unique uuid for the model to ensure there are no overwrites.
        val uniqueModelId =
          java.util.UUID.randomUUID().toString.replace("-", "")

        // Set a location to write the model to
        val modelDir = s"$baseDirectory${modelDescriptor}_$runId/$uniqueModelId"

        // log the model artifact
        saveModel(
          mlflowLoggingClient,
          modelDir,
          runId,
          x,
          modelDescriptor,
          uniqueModelId
        )

        // log the generation
        mlflowLoggingClient.logParam(runId, "generation", x.generation.toString)

        // Log custom tags if present
        if (_mlFlowCustomRunTags.nonEmpty) {
          logCustomTags(mlflowLoggingClient, runId, _mlFlowCustomRunTags)
        }

        /**
          * Set the remaining aspect of InferenceConfig for this run
          */
        // set the model save directory
        val inferencePath = inferenceSaveLocation.takeRight(1) match {
          case "/" => s"$inferenceSaveLocation$experimentId/"
          case _   => s"$inferenceSaveLocation/$experimentId/"
        }

        val inferenceLocation = inferencePath + runId

        val inferenceMlFlowConfig = getInternalMlFlowConfig(baseDirectory)
        val inferenceModelConfig = getInferenceModelConfig(
          modelFamily,
          modelType,
          "mlflow",
          inferenceMlFlowConfig,
          runId,
          modelDir)

        setInferenceModelConfig(inferenceModelConfig)
        setInferenceConfigStorageLocation(inferenceLocation)

        val inferenceConfig = getInferenceConfig

        val inferenceConfigAsJSON =
          convertInferenceConfigToJson(inferenceConfig)

        val inferenceConfigAsDF =
          convertInferenceConfigToDataFrame(inferenceConfig)

        //Save the inference config to the save location
        inferenceConfigAsDF.write.save(inferenceLocation)

        mlflowLoggingClient.setTag(
          runId,
          "InferenceConfig",
          inferenceConfigAsJSON.compactJson
        )

        mlflowLoggingClient.setTag(
          runId,
          "InferenceDataFrameLocation",
          inferenceLocation
        )

        withinRunId += 1

      }
    }

    MLFlowReturn(
      mlflowLoggingClient,
      experimentId,
      runIdPayloadBuffer.result().toArray
    )

  }

}
object MLFlowTracker {
  def apply(mlFlowConfig: MLFlowConfig): MLFlowTracker = {
    new MLFlowTracker()
    .setMlFlowTrackingURI(mlFlowConfig.mlFlowTrackingURI)
    .setMlFlowHostedAPIToken(mlFlowConfig.mlFlowAPIToken)
    .setMlFlowExperimentName(mlFlowConfig.mlFlowExperimentName)
    .setModelSaveDirectory(mlFlowConfig.mlFlowModelSaveDirectory)
    .setMlFlowLoggingMode(mlFlowConfig.mlFlowLoggingMode)
    .setMlFlowBestSuffix(mlFlowConfig.mlFlowBestSuffix)
  }
}
