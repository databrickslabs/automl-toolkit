package com.databricks.labs.automl.ensemble

import com.databricks.labs.automl.AutomationUnitTestsUtil.convertCsvToDf
import com.databricks.labs.automl.ensemble.setting.EnsembleSettingsBuilder
import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, last, trim}

class EnsembleTest extends AbstractUnitSpec {
/*
  it should "run stacking ensemble" in {
    val adultDf = convertCsvToDf("/adult_data.csv")
    var adultDfCleaned = adultDf
    for (colName <- adultDf.columns) {
      adultDfCleaned = adultDfCleaned
        .withColumn(
          colName.split("\\s+").mkString + "_trimmed",
          trim(col(colName))
        )
        .drop(colName)
    }
    adultDfCleaned = adultDfCleaned.withColumnRenamed("class_trimmed", "label")

    val overrides = Map(
      "labelCol" -> "label",
      "mlFlowLoggingFlag" -> false,
      "scalingFlag" -> true,
      "oneHotEncodeFlag" -> false,
      "numericBoundaries" -> Map(
        "numTrees" -> Tuple2(50.0, 100.0),
        "maxBins" -> Tuple2(10.0, 20.0),
        "maxDepth" -> Tuple2(2.0, 5.0),
        "minInfoGain" -> Tuple2(0.0, 0.03),
        "subSamplingRate" -> Tuple2(0.5, 1.0)
      ),
      "tunerParallelism" -> 4,
      "outlierFilterFlag" -> true,
      "outlierFilterPrecision" -> 0.05,
      "outlierLowerFilterNTile" -> 0.05,
      "outlierUpperFilterNTile" -> 0.95,
      "tunerTrainSplitMethod" -> "random", //TODO: ksample doesn't work with ensemble - need to move ksample stage out of FE
      "tunerKFold" -> 1,
      "tunerTrainPortion" -> 0.70,
      "tunerFirstGenerationGenePool" -> 5,
      "tunerNumberOfGenerations" -> 2,
      "tunerNumberOfParentsToRetain" -> 1,
      "tunerNumberOfMutationsPerGeneration" -> 1,
      "tunerGeneticMixing" -> 0.8,
      "tunerGenerationalMutationStrategy" -> "fixed",
      "tunerEvolutionStrategy" -> "batch",
      "pipelineDebugFlag" -> false
    )


    val metaOverrides = Map(
      "labelCol" -> "label",
      "mlFlowLoggingFlag" -> false,
      "scalingFlag" -> true,
      "oneHotEncodeFlag" -> false,
      "numericBoundaries" -> Map(
        "numTrees" -> Tuple2(50.0, 100.0),
        "maxBins" -> Tuple2(10.0, 20.0),
        "maxDepth" -> Tuple2(2.0, 5.0),
        "minInfoGain" -> Tuple2(0.0, 0.03),
        "subSamplingRate" -> Tuple2(0.5, 1.0)
      ),
      "tunerParallelism" -> 4,
      "outlierFilterFlag" -> true,
      "outlierFilterPrecision" -> 0.05,
      "outlierLowerFilterNTile" -> 0.05,
      "outlierUpperFilterNTile" -> 0.95,
      "tunerTrainSplitMethod" -> "random",
      "tunerKFold" -> 1,
      "tunerTrainPortion" -> 0.70,
      "tunerFirstGenerationGenePool" -> 5,
      "tunerNumberOfGenerations" -> 2,
      "tunerNumberOfParentsToRetain" -> 1,
      "tunerNumberOfMutationsPerGeneration" -> 1,
      "tunerGeneticMixing" -> 0.8,
      "tunerGenerationalMutationStrategy" -> "fixed",
      "tunerEvolutionStrategy" -> "batch",
      "pipelineDebugFlag" -> false,
      "covarianceFilterFlag" -> false,
      "outlierFilterFlag" -> false,
      "pearsonFilterFlag" -> false,
      "naFillFlag" -> false,
      "featureInteractionFlag" -> false,
      "scalingFlag" -> false,
      "dataPrepCachingFlag" -> false,
      "varianceFilterFlag" -> false,
      "fieldsToIgnoreInVector" -> adultDfCleaned.columns.filterNot(_.endsWith("_prediction"))
    )

    val randomForestConfig = ConfigurationGenerator
      .generateConfigFromMap("RandomForest", "classifier", overrides)

    val metaRandomForestConfig = ConfigurationGenerator
      .generateConfigFromMap("RandomForest", "classifier", metaOverrides)

    val tunerConfig = EnsembleSettingsBuilder
      .builder()
      .weakLearnersConfigs(Array(randomForestConfig))
      .metaLearnerConfig(Some(metaRandomForestConfig))
      .inputData(adultDfCleaned)
      .build()

    val r = EnsembleRunner.stacking(tunerConfig)

    r.get.bestEnsembleModel.transform(adultDfCleaned).show(10)


    // Test write and load of full inference pipeline
//    val pipelineSavePath = AutomationUnitTestsUtil
//      .getProjectDir() + "/target/pipeline-tests/infer-final-pipeline-lab"
//    runner
//      .bestPipelineModel("RandomForest")
//      .write
//      .overwrite()
//      .save(pipelineSavePath)
//    PipelineModel
//      .load(pipelineSavePath)
//      .transform(adultDfCleaned.drop("label"))
//      .show(100)

  }
*/
}
