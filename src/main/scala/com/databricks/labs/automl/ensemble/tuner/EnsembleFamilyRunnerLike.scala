package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.ensemble.setting.StackingEnsembleSettings
import com.databricks.labs.automl.executor.FamilyRunnerHelper
import com.databricks.labs.automl.executor.config.{ConfigurationGenerator, InstanceConfig}
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, FamilyFinalOutputWithPipeline, FamilyOutput, MainConfig}
import com.databricks.labs.automl.pipeline.{FeatureEngineeringOutput, FeatureEngineeringPipelineContext}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

case class EnsembleFamilyRunnerLikeReturnType(familyFinalOutputWithPipeline: FamilyFinalOutputWithPipeline,
                                              ensembleTunerSplits: Option[EnsembleTunerSplits])

case class MetaLearnerFamilyRunnerReturnType(familyFinalOutputWithPipeline: FamilyFinalOutputWithPipeline,
                                             weakLeanerPipelineModel: PipelineModel)

trait EnsembleFamilyRunnerLike[A, B] extends FamilyRunnerHelper {

  def execute(stackingEnsembleSettings: StackingEnsembleSettings,
              b: Option[B] = None): A

  def getMainConfigs(configs: Array[InstanceConfig]): Array[MainConfig] = {
    configs.map { x =>
      val mainConfiguration = ConfigurationGenerator.generateMainConfig(x)
      validatePerformanceSettings(
        mainConfiguration.geneticConfig.parallelism,
        mainConfiguration.modelFamily
      )
      mainConfiguration
    }
  }

  def getFePipelineModels(inputData: DataFrame,
                         mainConfigs: Array[MainConfig]): Array[FeatureEngineeringOutput] = {
    mainConfigs.map { mainConfiguration =>
      // Setup MLflow Run
      addMlFlowConfigForPipelineUse(mainConfiguration)
      var p: Option[FeatureEngineeringOutput] = None
//      Option[FeatureEngineeringOutput]
      withPipelineFailedException(mainConfiguration) {
        p = Some(FeatureEngineeringPipelineContext.generatePipelineModel(inputData, mainConfiguration))
      }
      p.get
    }
  }

  def runTuningAndGetOutput(fePipelineModels: Array[FeatureEngineeringOutput],
                            mainConfigs: Array[MainConfig],
                            configs: Array[InstanceConfig],
                            feDfWithAllModelFeatures: DataFrame,
                            testTrainSplits: Array[TrainSplitReferences]
                           ): (ArrayBuffer[FamilyOutput], mutable.Map[String, (FeatureEngineeringOutput, MainConfig)]) = {
    val outputBuffer = ArrayBuffer[FamilyOutput]()
    val pipelineConfigMap = scala.collection.mutable.Map[String, (FeatureEngineeringOutput, MainConfig)]()
    fePipelineModels
      .zipWithIndex
      .foreach {
        case (item: FeatureEngineeringOutput, i: Int) => {
          withPipelineFailedException(mainConfigs(i)) {
            val preppedDataOverride = DataGeneration(
              feDfWithAllModelFeatures,
              feDfWithAllModelFeatures.columns,
              item.decidedModel).copy(modelType = configs(i).predictionType)
            val output = TuningRunner.runTuning(mainConfigs(i), preppedDataOverride, testTrainSplits)
            outputBuffer += getNewFamilyOutPut(output, configs(i))
            pipelineConfigMap += configs(i).modelFamily -> (item, mainConfigs(i))
          }
        }
      }
    (outputBuffer, pipelineConfigMap)
  }
}