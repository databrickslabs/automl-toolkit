package com.databricks.spark.automatedml.tracking

import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.creds.BasicMlflowHostCreds

import scala.collection.mutable.Set
import scala.collection.JavaConversions._
import com.databricks.dbutils_v1.DBUtilsHolder.dbutils
import com.databricks.dbutils_v1.DBUtilsHolder.dbutils0
import com.databricks.spark.automatedml.params.GenericModelReturn
import org.apache.spark.ml.classification.RandomForestClassificationModel

import scala.collection.mutable

class MLFlowTracker {

  private var _mlFlowTrackingURI: String = _
  private var _mlFlowExperimentName: String = "default"
  private var _mlFlowHostedAPIToken: String = _


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

  //Intentionally not providing a getter for an API token.

  def getMlFlowTrackingURI: String = _mlFlowTrackingURI
  def getMlFlowExperimentName: String = _mlFlowExperimentName

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

  def logMlFlowDataAndModels(runData: Array[GenericModelReturn]) = {

    val mlflowLoggingClient = createHostedMlFlowClient()

    val generationSet = mutable.Set[Int]()
    runData.map(x => generationSet += x.generation)
    val uniqueGenerations = generationSet.result.toArray.sortWith(_<_)

    // loop through each generation and log the data

    uniqueGenerations.foreach{g =>

      // create a new MlFlowRun
      val runId = generateMlFlowRun(mlflowLoggingClient, g.toString)

      // get the runs from this generation
      val currentGen = runData.filter(x => x.generation == g)

      currentGen.foreach{x =>

        val hyperParamKeys = x.hyperParams.keys

        // TODO: make this a function!
        hyperParamKeys.foreach{k =>
          val valueData = x.hyperParams.get(k)
          mlflowLoggingClient.logParam(runId, k, valueData.toString)
        }
        val metricKeys = x.metrics.keys

        metricKeys.foreach{k =>
          val valueData = x.metrics.get(k)
          mlflowLoggingClient.logMetric(runId, k, valueData.toString.toDouble)
        }

        // log the model family
       //TODO: this will REQUIRE saving to BlobStore, then mounting a Fuse Point to it, then passing that ref to mlflow.
        // need a method that will generate the fuse mount point
        // create a save path generator with a base location and the experiment name + run id (generation) + a uuid?

        x.model.asInstanceOf[RandomForestClassificationModel].write.overwrite().save()

        /**
          *
          *
          * From Andre:
          *
          * def saveModelAsSparkMl(runId: String, baseModelDir: String, model: PipelineModel) = {
          * val modelDir = s"$baseModelDir/spark_model"
          * //model.save(modelDir)
          *   model.write.overwrite().save(modelDir) // hangs if we pass a Fuse path
          *   mlflowClient.logArtifacts(runId, new File(mkFusePath(modelDir)), "spark_model")
          *   mlflowClient.setTag(runId, "LocalPath_SparkModel", modelDir)
          * }
          *
          *
          */
        // log the model artifact

        // log the generation
        mlflowLoggingClient.logParam(runId, "generation", x.toString)


      }






    }




  }





}
