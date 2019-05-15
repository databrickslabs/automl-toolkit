package com.databricks.labs.automl.executor

import com.databricks.labs.automl.AutomationRunner
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig}
import com.databricks.labs.automl.params._
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer




class FamilyRunner(data: DataFrame, configs: Array[InstanceConfig]) extends SparkSessionWrapper{



  //TODO: move this method to companion object

  private def augmentDF(modelType: String, dataFrame: DataFrame): DataFrame = {

    dataFrame.withColumn("model", lit(modelType))

  }

  //TODO: move this.
  private def unifyFamilyOutput(outputArray: Array[FamilyOutput]): FamilyFinalOutput = {

    var modelReport = ArrayBuffer[GroupedModelReturn]()
    var generationReport = ArrayBuffer[GenerationalReport]()
    var modelReportDataFrame = spark.emptyDataFrame
    var generationReportDataFrame = spark.emptyDataFrame

    outputArray.map{ x =>

      x.modelReport.map{ y =>
        modelReport += GroupedModelReturn(
          modelFamily = x.modelType,
          hyperParams = y.hyperParams,
          model = y.model,
          score = y.score,
          metrics = y.metrics,
          generation = y.generation
        )
      }
      generationReport +: x.generationReport
      modelReportDataFrame.union(x.modelReportDataFrame)
      generationReportDataFrame.union(x.generationReportDataFrame)
    }

    // TODO: VALIDATE THAT THIS WORKS PROPERLY!!!!

    FamilyFinalOutput(
      modelReport = modelReport.toArray,
      generationReport = generationReport.toArray,
      modelReportDataFrame = modelReportDataFrame,
      generationReportDataFrame= generationReportDataFrame
    )

  }

  /**
    * For now, until a refactor of Automation Runner, will be using MainConfig object configurations.
    */

  def execute(): FamilyFinalOutput = {

    val outputBuffer = ArrayBuffer[FamilyOutput]()

    configs.foreach{ x =>

      val mainConfiguration = ConfigurationGenerator.generateMainConfig(x)

      val runner: AutomationRunner = new AutomationRunner(data)
        .setMainConfig(mainConfiguration)

      val preppedData = runner.prepData()

      val preppedDataOverride = preppedData.copy(modelType = x.predictionType)

      val output = runner.executeTuning(preppedDataOverride)

      outputBuffer += new FamilyOutput(x.modelFamily){
        override def modelReport: Array[GenericModelReturn] = output.modelReport
        override def generationReport: Array[GenerationalReport] = output.generationReport
        override def modelReportDataFrame: DataFrame = augmentDF(x.modelFamily, output.modelReportDataFrame)
        override def generationReportDataFrame: DataFrame = augmentDF(x.modelFamily, output.generationReportDataFrame)
      }
    }

    unifyFamilyOutput(outputBuffer.toArray)

  }




  // Method listing:
  // 1. DataPrep / fillna operations
  // 2. cache if requested
  // 3. set the configuration for different models
  // 4. execute the modeling runs
  // 5. log all to mlflow
  // 6.



}
