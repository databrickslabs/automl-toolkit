package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, StringIndexerModel}
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer

class XgboostLoanRiskTest extends AbstractUnitSpec {

  ignore should "run successfully" in {
    val loanRiskDf = AutomationUnitTestsUtil.convertCsvToDf("/ml_msd.csv").withColumn("decade", col("decade")+lit(""))
//    val genericMapOverrides = Map(
//      "labelCol" -> "label",
//      "scoringMetric" -> "areaUnderROC",
//      "oneHotEncodeFlag" -> true,
//      "autoStoppingFlag" -> true,
//      "tunerAutoStoppingScore" -> 0.91,
//      "tunerParallelism" -> 1 * 2,
//      "tunerKFold" -> 2,
//      "tunerTrainPortion" -> 0.7,
//      "tunerTrainSplitMethod" -> "stratified",
//      "tunerInitialGenerationMode" -> "permutations",
//      "tunerInitialGenerationPermutationCount" -> 8,
//      "tunerInitialGenerationIndexMixingMode" -> "linear",
//      "tunerInitialGenerationArraySeed" -> 42L,
//      "tunerFirstGenerationGenePool" -> 16,
//      "tunerNumberOfGenerations" -> 3,
//      "tunerNumberOfParentsToRetain" -> 2,
//      "tunerNumberOfMutationsPerGeneration" -> 4,
//      "tunerGeneticMixing" -> 0.8,
//      "tunerGenerationalMutationStrategy" -> "fixed",
//      "tunerEvolutionStrategy" -> "batch",
//      "tunerHyperSpaceInferenceFlag" -> true,
//      "tunerHyperSpaceInferenceCount" -> 20000,
//      "tunerHyperSpaceModelType" -> "XGBoost",
//      "tunerHyperSpaceModelCount" -> 8,
//      "mlFlowLoggingFlag" -> false,
//      "mlFlowLogArtifactsFlag" -> false,
//      "pipelineDebugFlag" -> true
//    )

    val rfNumericBoundaries = Map(
      "numTrees" -> Tuple2(50.0, 1000.0),
      "maxBins" -> Tuple2(2.0, 10.0),
      "maxDepth" -> Tuple2(2.0, 20.0),
      "minInfoGain" -> Tuple2(0.0, 1.0),
      "subSamplingRate" -> Tuple2(0.5, 1.0)
    )

    val genericClassifierOverrides = Map(
      "labelCol" -> "decade",
      "scoringMetric" -> "f1", //weightedPrecision
      "naFillFlag" -> true,
      "varianceFilterFlag" -> false,
      "outlierFilterFlag" -> false, // TODO
      "pearsonFilterFlag" -> false, // TO REVIEW
      "covarianceFilterFlag" -> false,
      "oneHotEncodeFlag" -> false,
      "scalingFlag" -> false,
      "dataPrepCachingFlag" -> false,
      "numericBoundaries" -> rfNumericBoundaries,
      "autoStoppingFlag" -> false,
      "tunerAutoStoppingScore" -> 0.91,
      "tunerParallelism" -> 8,
      "tunerKFold" -> 2,
      "tunerTrainPortion" -> 0.7,
      "tunerTrainSplitMethod" -> "stratified",
      "tunerInitialGenerationMode" -> "permutations",
      "tunerInitialGenerationPermutationCount" -> 8,
      "tunerInitialGenerationIndexMixingMode" -> "linear",
      "tunerInitialGenerationArraySeed" -> 42L,
      "tunerFirstGenerationGenePool" -> 16,
      "tunerNumberOfGenerations" -> 3,
      "tunerNumberOfParentsToRetain" -> 2,
      "tunerNumberOfMutationsPerGeneration" -> 4,
      "tunerGeneticMixing" -> 0.8,
      "tunerGenerationalMutationStrategy" -> "fixed",
      "tunerEvolutionStrategy" -> "batch",
      "tunerHyperSpaceInferenceFlag" -> true,
      "tunerHyperSpaceInferenceCount" -> 400000,
//      "tunerHyperSpaceModelType" -> "XGBoost",
      "tunerHyperSpaceModelCount" -> 8,
      "mlFlowLoggingFlag" -> false,
      "mlFlowLogArtifactsFlag" -> false,
      "fillConfigCardinalityLimit" -> 1000000
    )


    val xgBoostConfig = ConfigurationGenerator.generateConfigFromMap(
      "randomforest",
      "classifier",
      genericClassifierOverrides
    )
    val familyRunner =
      FamilyRunner(loanRiskDf, Array(xgBoostConfig)).executeWithPipeline()
    familyRunner.bestPipelineModel(("XGBoost")).transform(loanRiskDf).show(10)
  }

/*
  "ad" should "" in {


    val loanRiskDf = AutomationUnitTestsUtil.convertCsvToDf("/loan_risk.csv")
    loanRiskDf.show(10)

    val si = new StringIndexer().setInputCol("home_ownership").setOutputCol("home_ownership_si")
    val ohe = new OneHotEncoder().setInputCol("home_ownership_si").setOutputCol("home_ownership_si_ohe")

    val siI = new StringIndexer().setInputCol("purpose").setOutputCol("purpose_si")
    val oheI = new OneHotEncoder().setInputCol("purpose_si").setOutputCol("purpose_si_ohe")

    val innerPipeline = new Pipeline().setStages(Array(siI, oheI))

    val outerPipelineModel = new Pipeline().setStages(Array(si, ohe, innerPipeline)).fit(loanRiskDf)

    outerPipelineModel.transform(loanRiskDf).show(10)


    val oo =  getStringIndexerMapping(outerPipelineModel)

    print("done")

  }
*/
  def getStringIndexerMapping(
                               pipeline: PipelineModel
                             ): Array[StringIndexerMappings] = {
    val siStages = pipeline.stages
      .collect {
        case x: StringIndexerModel =>
          val indexer = x.asInstanceOf[StringIndexerModel]
          StringIndexerMappings(indexer.getInputCol, indexer.getOutputCol)
        case x: PipelineModel => getStringIndexerMapping(x)
      }

    siStages.asInstanceOf[Array[StringIndexerMappings]]
  }

//  def getStringIndexerMapping(
//                               pipeline: PipelineModel,
//                               siStages: ArrayBuffer[StringIndexerMappings] = ArrayBuffer[StringIndexerMappings]()
//                             ): ArrayBuffer[StringIndexerMappings] = {
//   pipeline.stages
//      .collect {
//        case x: StringIndexerModel =>
//          val indexer = x.asInstanceOf[StringIndexerModel]
//          siStages += StringIndexerMappings(indexer.getInputCol, indexer.getOutputCol)
//        case x: PipelineModel => getStringIndexerMapping(x, siStages)
//      }
//
//    siStages
//  }

  case class StringIndexerMappings(inputCol: String, outputCol: String)

}
