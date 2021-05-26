package com.databricks.labs.automl.model

import com.databricks.labs.automl.executor.DataPrep
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.model.tools.split.DataSplitUtility
import com.databricks.labs.automl.params.LogisticRegressionModelsWithResults
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class LogisticRegressionTunerTest extends AbstractUnitSpec {

  "LogisticRegressionTuner" should "throw IllegalArgumentException for passing invalid params" in {
    a[IllegalArgumentException] should be thrownBy {
      val data =
        new DataPrep(AutomationUnitTestsUtil.getAdultDf()).prepData().data
      val trainSplits = DataSplitUtility.split(
        data,
        1,
        "random",
        "label",
        "dbfs:/test",
        "cache",
        "LogisticRegression",
        2,
        0.7,
        "synth",
        "datetime",
        0.02,
        0.6
      )

      new LogisticRegressionTuner(null, trainSplits).evolveBest()
    }
  }

  it should "should throw IllegalArgumentException for passing invalid dataset" in {
    a[IllegalArgumentException] should be thrownBy {

      val data =
        new DataPrep(AutomationUnitTestsUtil.getAdultDf()).prepData().data
      val trainSplits = DataSplitUtility.split(
        data,
        1,
        "random",
        "label",
        "dbfs:/test",
        "cache",
        "LogisticRegression",
        2,
        0.7,
        "synth",
        "datetime",
        0.02,
        0.6
      )

      new LogisticRegressionTuner(
        AutomationUnitTestsUtil.sparkSession.emptyDataFrame,
        trainSplits
      ).evolveBest()
    }
  }

  it should "should return valid Binary Classification Model" in {

    val _mainConfig = ConfigurationGenerator.generateMainConfig(
      ConfigurationGenerator
        .generateDefaultConfig("logisticregression", "classifier")
    )

    val data =
      new DataPrep(AutomationUnitTestsUtil.getAdultDf()).prepData().data
    val trainSplits = DataSplitUtility.split(
      data,
      1,
      "random",
      _mainConfig.labelCol,
      "dbfs:/test",
      "cache",
      "LogisticRegression",
      2,
      0.7,
      "synth",
      "datetime",
      0.02,
      0.6
    )

    val logisticRegressionModelsWithResults
      : LogisticRegressionModelsWithResults =
      new LogisticRegressionTuner(data, trainSplits)
        .setFirstGenerationGenePool(5)
        .setLabelCol(_mainConfig.labelCol)
        .setFeaturesCol(_mainConfig.featuresCol)
        .setFieldsToIgnore(_mainConfig.fieldsToIgnoreInVector)
        .setLogisticRegressionNumericBoundaries(
          Map(
            "elasticNetParams" -> Tuple2(1.0, 1.0),
            "maxIter" -> Tuple2(10.0, 20.0),
            "regParam" -> Tuple2(0.9, 1.0),
            "tolerance" -> Tuple2(1E-9, 1E-5)
          )
        )
        .setScoringMetric("areaUnderROC")
        .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
        .setTrainSplitMethod(_mainConfig.geneticConfig.trainSplitMethod)
        .setSyntheticCol(_mainConfig.geneticConfig.kSampleConfig.syntheticCol)
        .setKGroups(_mainConfig.geneticConfig.kSampleConfig.kGroups)
        .setKMeansMaxIter(_mainConfig.geneticConfig.kSampleConfig.kMeansMaxIter)
        .setKMeansTolerance(
          _mainConfig.geneticConfig.kSampleConfig.kMeansTolerance
        )
        .setKMeansDistanceMeasurement(
          _mainConfig.geneticConfig.kSampleConfig.kMeansDistanceMeasurement
        )
        .setKMeansSeed(_mainConfig.geneticConfig.kSampleConfig.kMeansSeed)
        .setKMeansPredictionCol(
          _mainConfig.geneticConfig.kSampleConfig.kMeansPredictionCol
        )
        .setLSHHashTables(_mainConfig.geneticConfig.kSampleConfig.lshHashTables)
        .setLSHSeed(_mainConfig.geneticConfig.kSampleConfig.lshSeed)
        .setLSHOutputCol(_mainConfig.geneticConfig.kSampleConfig.lshOutputCol)
        .setQuorumCount(_mainConfig.geneticConfig.kSampleConfig.quorumCount)
        .setMinimumVectorCountToMutate(
          _mainConfig.geneticConfig.kSampleConfig.minimumVectorCountToMutate
        )
        .setVectorMutationMethod(
          _mainConfig.geneticConfig.kSampleConfig.vectorMutationMethod
        )
        .setMutationMode(_mainConfig.geneticConfig.kSampleConfig.mutationMode)
        .setMutationValue(_mainConfig.geneticConfig.kSampleConfig.mutationValue)
        .setLabelBalanceMode(
          _mainConfig.geneticConfig.kSampleConfig.labelBalanceMode
        )
        .setCardinalityThreshold(
          _mainConfig.geneticConfig.kSampleConfig.cardinalityThreshold
        )
        .setNumericRatio(_mainConfig.geneticConfig.kSampleConfig.numericRatio)
        .setNumericTarget(_mainConfig.geneticConfig.kSampleConfig.numericTarget)
        .setTrainSplitChronologicalColumn(
          _mainConfig.geneticConfig.trainSplitChronologicalColumn
        )
        .setTrainSplitChronologicalRandomPercentage(
          _mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage
        )
        .setParallelism(4)
        .setKFold(1)
        .setSeed(_mainConfig.geneticConfig.seed)
        .setOptimizationStrategy("maximize")
        .setFirstGenerationGenePool(5)
        .setNumberOfMutationGenerations(2)
        .setNumberOfMutationsPerGeneration(5)
        .setNumberOfParentsToRetain(2)
        .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
        .setGenerationalMutationStrategy(
          _mainConfig.geneticConfig.generationalMutationStrategy
        )
        .setMutationMagnitudeMode(
          _mainConfig.geneticConfig.mutationMagnitudeMode
        )
        .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
        .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
        .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
        .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
        .setContinuousEvolutionImprovementThreshold(
          _mainConfig.geneticConfig.continuousEvolutionImprovementThreshold
        )
        .setGeneticMBORegressorType(
          _mainConfig.geneticConfig.geneticMBORegressorType
        )
        .setGeneticMBOCandidateFactor(
          _mainConfig.geneticConfig.geneticMBOCandidateFactor
        )
        .setDataReductionFactor(_mainConfig.dataReductionFactor)
        .setFirstGenMode(_mainConfig.geneticConfig.initialGenerationMode)
        .setFirstGenPermutations(4)
        .setFirstGenIndexMixingMode(
          _mainConfig.geneticConfig.initialGenerationConfig.indexMixingMode
        )
        .setFirstGenArraySeed(
          _mainConfig.geneticConfig.initialGenerationConfig.arraySeed
        )
        .setHyperSpaceModelCount(50000)
        .evolveBest()
    assert(
      logisticRegressionModelsWithResults != null,
      "logisticRegressionModelsWithResults should not have been null"
    )
    assert(
      logisticRegressionModelsWithResults.evalMetrics != null,
      "evalMetrics should not have been null"
    )
    assert(
      logisticRegressionModelsWithResults.model != null,
      "model should not have been null"
    )
    assert(
      logisticRegressionModelsWithResults.modelHyperParams != null,
      "modelHyperParams should not have been null"
    )
  }


}
