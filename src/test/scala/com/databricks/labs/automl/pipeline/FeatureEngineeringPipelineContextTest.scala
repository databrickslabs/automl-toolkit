package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.{AbstractUnitSpec, PipelineTestUtils}

class FeatureEngineeringPipelineContextTest extends AbstractUnitSpec {

  "FeatureEngineeringPipelineContextTest" should "correctly generate feature engineered dataset" in {
    val testVars = PipelineTestUtils.getTestVars()

    // Generate config
    val overrides = Map("labelCol" -> "label", "mlFlowLoggingFlag" -> false, "scalingFlag" -> true, "oneHotEncodeFlag" -> true)
    val randomForestConfig = ConfigurationGenerator.generateConfigFromMap("RandomForest", "classifier", overrides)
    randomForestConfig.switchConfig.outlierFilterFlag = true
    randomForestConfig.featureEngineeringConfig.outlierFilterPrecision = 0.05
    randomForestConfig.featureEngineeringConfig.outlierLowerFilterNTile = 0.05
    randomForestConfig.featureEngineeringConfig.outlierUpperFilterNTile = 0.95
    randomForestConfig.tunerConfig.tunerParallelism = 10
    randomForestConfig.tunerConfig.tunerTrainSplitMethod = "kSample"
    val modelMainConfig = ConfigurationGenerator.generateMainConfig(randomForestConfig)
    val featuresEngPipelineModel = FeatureEngineeringPipelineContext
      .generatePipelineModel(testVars.df, modelMainConfig).pipelineModel

    PipelineTestUtils
      .saveAndLoadPipelineModel(featuresEngPipelineModel, testVars.df, "full-feature-eng-pipeline")
      .transform(testVars.df)
      .show(10)
  }

}
