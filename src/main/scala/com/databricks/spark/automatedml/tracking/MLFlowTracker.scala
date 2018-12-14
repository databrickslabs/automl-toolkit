package com.databricks.spark.automatedml.tracking

import java.io.File

import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.creds.BasicMlflowHostCreds

import com.databricks.spark.automatedml.params.GenericModelReturn
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, GBTRegressionModel, LinearRegressionModel, RandomForestRegressionModel}

import scala.collection.mutable

class MLFlowTracker {


  private var _mlFlowTrackingURI: String = _
  private var _mlFlowExperimentName: String = "default"
  private var _mlFlowHostedAPIToken: String = _
  private var _modelSaveDirectory: String = _


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

  //Intentionally not providing a getter for an API token.

  def getMlFlowTrackingURI: String = _mlFlowTrackingURI
  def getMlFlowExperimentName: String = _mlFlowExperimentName
  def getModelSaveDirectory: String = _modelSaveDirectory

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
      //TODO: if we can get the API token from here automatically, do it.
      //val token = dbutils.notebook.getContext()
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

  private def createFusePath(path: String): String = {
    path.replace("dbfs:", "/dbfs")
  }

  private def saveModel(client: MlflowClient, path: String, runId: String, modelReturn: GenericModelReturn,
                        modelDescriptor: String, modelId: String): Unit = {

    modelDescriptor match {
      case "regressor_RandomForest" =>
        modelReturn.model.asInstanceOf[RandomForestRegressionModel].write.overwrite().save(path)
        client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, s"SparkModel_$modelId", path)
      case "classifier_RandomForest" =>
        modelReturn.model.asInstanceOf[RandomForestClassificationModel].write.overwrite().save(path)
        client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, s"SparkModel_$modelId", path)
      case "regressor_GBT" =>
        modelReturn.model.asInstanceOf[GBTRegressionModel].write.overwrite().save(path)
        client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, s"SparkModel_$modelId", path)
      case "classifier_GBT" =>
        modelReturn.model.asInstanceOf[GBTClassificationModel].write.overwrite().save(path)
        client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, s"SparkModel_$modelId", path)
      case "classifier_MLPC" =>
        modelReturn.model.asInstanceOf[MultilayerPerceptronClassificationModel].write.overwrite().save(path)
        client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, s"SparkModel_$modelId", path)
      case "regressor_LinearRegression" =>
        modelReturn.model.asInstanceOf[LinearRegressionModel].write.overwrite().save(path)
        client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, s"SparkModel_$modelId", path)
      case "classifer_LogisticRegression" =>
        modelReturn.model.asInstanceOf[LogisticRegressionModel].write.overwrite().save(path)
        client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, s"SparkModel_$modelId", path)
      case "regressor_SVM" =>
        modelReturn.model.asInstanceOf[LinearSVCModel].write.overwrite().save(path)
        client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, s"SparkModel_$modelId", path)
      case "regressor_Trees" =>
        modelReturn.model.asInstanceOf[DecisionTreeRegressionModel].write.overwrite().save(path)
        client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, s"SparkModel_$modelId", path)
      case "classifier_Trees" =>
        modelReturn.model.asInstanceOf[DecisionTreeClassificationModel].write.overwrite().save(path)
        client.logArtifacts(runId, new File(createFusePath(path)))
        client.setTag(runId, s"SparkModel_$modelId", path)
      case _ => throw new UnsupportedOperationException(
        s"Model Type $modelDescriptor is not supported for mlflow logging.")
    }
  }


  def logMlFlowDataAndModels(runData: Array[GenericModelReturn], modelFamily: String, modelType: String): Unit = {

    val mlflowLoggingClient = createHostedMlFlowClient()

    val generationSet = mutable.Set[Int]()
    runData.map(x => generationSet += x.generation)
    val uniqueGenerations = generationSet.result.toArray.sortWith(_<_)

    // set the model save directory
    val baseDirectory = _modelSaveDirectory.takeRight(1) match {
      case "/" => s"${_modelSaveDirectory}/${_mlFlowExperimentName}/"
      case _ => s"${_modelSaveDirectory}${_mlFlowExperimentName}/"
    }

    val modelDescriptor = s"${modelType}_$modelFamily"

    // loop through each generation and log the data
    uniqueGenerations.foreach{g =>

      // create a new MlFlowRun
      val runId = generateMlFlowRun(mlflowLoggingClient, g.toString)

      // get the runs from this generation
      val currentGen = runData.filter(x => x.generation == g)

      currentGen.foreach{x =>

        val hyperParamKeys = x.hyperParams.keys

        hyperParamKeys.foreach{k =>
          val valueData = x.hyperParams.get(k)
          mlflowLoggingClient.logParam(runId, k, valueData.toString)
        }
        val metricKeys = x.metrics.keys

        metricKeys.foreach{k =>
          val valueData = x.metrics.get(k)
          mlflowLoggingClient.logMetric(runId, k, valueData.toString.toDouble)
        }

        // Generate a new unique uuid for the model to ensure there are no overwrites.
        val uniqueModelId = java.util.UUID.fromString(x.hyperParams.toString())

        // Set a location to write the model to
        val modelDir = s"$baseDirectory/${modelDescriptor}_$runId/$uniqueModelId"

        // log the model artifact
        saveModel(mlflowLoggingClient, modelDir, runId, x, modelDescriptor, uniqueModelId.toString)

        // log the generation
        mlflowLoggingClient.logParam(runId, "generation", x.toString)

      }
    }
  }





}
