package com.databricks.spark.automatedml.tracking

import java.io.File



import org.mlflow.tracking.MlflowClient
//import org.mlflow.tracking.creds.BasicMlflowHostCreds
import org.mlflow.tracking.creds._
import scala.collection.JavaConversions._

import com.databricks.spark.automatedml.params.GenericModelReturn
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, GBTRegressionModel, LinearRegressionModel, RandomForestRegressionModel}

import scala.collection.mutable

class MLFlowTracker {


  private var _mlFlowTrackingURI: String = _
  private var _mlFlowExperimentName: String = "default"
  private var _mlFlowHostedAPIToken: String = _
  private var _modelSaveDirectory: String = _
  private var _logArtifacts: Boolean = false


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

  //Intentionally not providing a getter for an API token.

  def getMlFlowTrackingURI: String = _mlFlowTrackingURI
  def getMlFlowExperimentName: String = _mlFlowExperimentName
  def getModelSaveDirectory: String = _modelSaveDirectory
  def getArtifactLogSetting: Boolean = _logArtifacts

  /**
    * Method for either getting an existing experiment by name, or creating a new one by name and returning the id
    * @param client: MlflowClient to get access to the mlflow service agent
    * @return the experiment id from either an existing run or the newly created one.
    */

  private def getOrCreateExperimentId(client: MlflowClient): Long = {

    val experiment = client.getExperimentByName(_mlFlowExperimentName)
    if(experiment.isPresent) experiment.get().getExperimentId else client.createExperiment(_mlFlowExperimentName)

  }

  //
  private def createHostedMlFlowClient(): MlflowClient = {

    val hosted: Boolean = _mlFlowTrackingURI.contains("databricks.com")

    if (hosted) {
      new MlflowClient(new BasicMlflowHostCreds(_mlFlowTrackingURI, _mlFlowHostedAPIToken))
    } else {
      new MlflowClient(_mlFlowTrackingURI)
    }
  }

  /**
    * Method for generating an entry to log to for the
    * @param runIdentifier
    * @return :(MlflowClient, String) The client logging object and the runId uuid for use in logging.
    */
  private def generateMlFlowRun(client: MlflowClient, runIdentifier: String): String = {

    val experimentId = getOrCreateExperimentId(client)

    val runId = client.createRun(experimentId, runIdentifier).getRunUuid

    runId

  }

  import org.mlflow.api.proto.Service.CreateRun
  import org.mlflow.tracking.MlflowClient

  private def generateMlFlowRun(client: MlflowClient, experimentID: Long, runIdentifier: String,
                                runName: String, sourceVer: String): String = {

    val request: CreateRun.Builder = CreateRun.newBuilder()
      .setExperimentId(experimentID)
      .setRunName(runName)
      .setSourceVersion(sourceVer)
      .setSourceName(runIdentifier)
      .setStartTime(System.currentTimeMillis())

    val run = client.createRun(request.build())

    run.getRunUuid
  }

  private def createFusePath(path: String): String = {
    path.replace("dbfs:", "/dbfs")
  }

  /**
    * Private method for saving an individual model, creating a Fuse mount for it, and registering the artifact.
    * @param client MlFlow client that has been registered.
    * @param path Path in blob store for saving the SparkML Model
    * @param runId Unique runID for the run to log model artifacts to.
    * @param modelReturn Modeling payload for the run in order to extract the specific model type
    * @param modelDescriptor Text Assignment for the model family + type of model that was run
    * @param modelId Unique uuid identifier for the model.
    */
  private def saveModel(client: MlflowClient, path: String, runId: String, modelReturn: GenericModelReturn,
                        modelDescriptor: String, modelId: String): Unit = {

    modelDescriptor match {
            case "regressor_RandomForest" =>
              modelReturn.model.asInstanceOf[RandomForestRegressionModel].write.overwrite().save(path)
              if(_logArtifacts) client.logArtifacts(runId, new File(createFusePath(path)))
              client.setTag(runId, s"SparkModel_$modelId", path)
            case "classifier_RandomForest" =>
              modelReturn.model.asInstanceOf[RandomForestClassificationModel].write.overwrite().save(path)
              if(_logArtifacts) client.logArtifacts(runId, new File(createFusePath(path)))
              client.setTag(runId, s"SparkModel_$modelId", path)
            case "regressor_GBT" =>
              modelReturn.model.asInstanceOf[GBTRegressionModel].write.overwrite().save(path)
              if(_logArtifacts) client.logArtifacts(runId, new File(createFusePath(path)))
              client.setTag(runId, s"SparkModel_$modelId", path)
            case "classifier_GBT" =>
              modelReturn.model.asInstanceOf[GBTClassificationModel].write.overwrite().save(path)
              if(_logArtifacts) client.logArtifacts(runId, new File(createFusePath(path)))
              client.setTag(runId, s"SparkModel_$modelId", path)
            case "classifier_MLPC" =>
              modelReturn.model.asInstanceOf[MultilayerPerceptronClassificationModel].write.overwrite().save(path)
              if(_logArtifacts) client.logArtifacts(runId, new File(createFusePath(path)))
              client.setTag(runId, s"SparkModel_$modelId", path)
            case "regressor_LinearRegression" =>
              modelReturn.model.asInstanceOf[LinearRegressionModel].write.overwrite().save(path)
              if(_logArtifacts) client.logArtifacts(runId, new File(createFusePath(path)))
              client.setTag(runId, s"SparkModel_$modelId", path)
            case "classifer_LogisticRegression" =>
              modelReturn.model.asInstanceOf[LogisticRegressionModel].write.overwrite().save(path)
              if(_logArtifacts) client.logArtifacts(runId, new File(createFusePath(path)))
              client.setTag(runId, s"SparkModel_$modelId", path)
            case "regressor_SVM" =>
              modelReturn.model.asInstanceOf[LinearSVCModel].write.overwrite().save(path)
              if(_logArtifacts) client.logArtifacts(runId, new File(createFusePath(path)))
              client.setTag(runId, s"SparkModel_$modelId", path)
            case "regressor_Trees" =>
              modelReturn.model.asInstanceOf[DecisionTreeRegressionModel].write.overwrite().save(path)
              if(_logArtifacts) client.logArtifacts(runId, new File(createFusePath(path)))
              client.setTag(runId, s"SparkModel_$modelId", path)
            case "classifier_Trees" =>
              modelReturn.model.asInstanceOf[DecisionTreeClassificationModel].write.overwrite().save(path)
              if(_logArtifacts) client.logArtifacts(runId, new File(createFusePath(path)))
              client.setTag(runId, s"SparkModel_$modelId", path)
            case _ => throw new UnsupportedOperationException(
              s"Model Type $modelDescriptor is not supported for mlflow logging.")
          }
  }

  /**
    * Public method for logging a model, parameters, and metrics to MlFlow
    * @param runData Full collection parameters, results, and models for the automatedML experiment
    * @param modelFamily Type of Model Family used (e.g. "RandomForest")
    * @param modelType Type of Model used (e.g. "regression")
    */
  def logMlFlowDataAndModels(runData: Array[GenericModelReturn], modelFamily: String, modelType: String): Unit = {

    val mlflowLoggingClient = createHostedMlFlowClient()

    val experimentId = getOrCreateExperimentId(mlflowLoggingClient)

    var totalVersion = mlflowLoggingClient.getExperiment(experimentId).getRunsCount

    val generationSet = mutable.Set[Int]()
    runData.map(x => generationSet += x.generation)
    val uniqueGenerations = generationSet.result.toArray.sortWith(_<_)

    // set the model save directory
    val baseDirectory = _modelSaveDirectory.takeRight(1) match {
      case "/" => s"${_modelSaveDirectory}${_mlFlowExperimentName}/"
      case _ => s"${_modelSaveDirectory}/${_mlFlowExperimentName}/"
    }

    val modelDescriptor = s"${modelType}_$modelFamily"

    // loop through each generation and log the data
    uniqueGenerations.foreach{g =>

      // get the runs from this generation
      val currentGen = runData.filter(x => x.generation == g)

      var withinRunId = 0

      currentGen.foreach{x =>

        // create a new MlFlowRun
        //val runId = generateMlFlowRun(mlflowLoggingClient, g.toString)

        totalVersion += 1

        val uniqueRunIdent = s"${modelFamily}_${modelType}_${x.generation.toString}_${withinRunId.toString}_${
          x.score.toString}"

        val runName = "run_" + x.generation.toString + "_" + withinRunId.toString

        val runId = generateMlFlowRun(mlflowLoggingClient, experimentId, uniqueRunIdent, runName,
          totalVersion.toString)

        val hyperParamKeys = x.hyperParams.keys

        hyperParamKeys.foreach{k =>
          val valueData = x.hyperParams(k)
          mlflowLoggingClient.logParam(runId, k, valueData.toString)
        }
        val metricKeys = x.metrics.keys

        metricKeys.foreach{k =>
          val valueData = x.metrics(k)
          mlflowLoggingClient.logMetric(runId, k, valueData.toString.toDouble)
        }

        // Generate a new unique uuid for the model to ensure there are no overwrites.
        val uniqueModelId = java.util.UUID.randomUUID().toString.replace("-", "")

        // Set a location to write the model to
        val modelDir = s"$baseDirectory${modelDescriptor}_$runId/$uniqueModelId"

        // log the model artifact
        saveModel(mlflowLoggingClient, modelDir, runId, x, modelDescriptor, uniqueModelId)

        // log the generation
        mlflowLoggingClient.logParam(runId, "generation", x.generation.toString)

        withinRunId += 1

      }
    }
  }





}
