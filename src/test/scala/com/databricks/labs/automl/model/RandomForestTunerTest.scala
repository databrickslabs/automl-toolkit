package com.databricks.labs.automl.model

import com.databricks.labs.automl.executor.DataPrep
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.model.tools.split.DataSplitUtility
import com.databricks.labs.automl.params.RandomForestModelsWithResults
import com.databricks.labs.automl.{
  AbstractUnitSpec,
  AutomationUnitTestsUtil,
  DiscreteTestDataGenerator
}

class RandomForestTunerTest extends AbstractUnitSpec {

  "RandomForestTuner" should "throw UnsupportedOperationException for passing invalid params" in {
    a[UnsupportedOperationException] should be thrownBy {
      val splitData = DataSplitUtility.split(
        AutomationUnitTestsUtil.getAdultDf(),
        1,
        "random",
        "label",
        "dbfs:/test",
        "cache",
        "RandomForest",
        2,
        0.7,
        "synth",
        "datetime",
        0.02,
        0.6
      )

      new RandomForestTuner(null, splitData, null).evolveBest()
    }
  }

  it should "throw UnsupportedOperationException for passing invalid modelSelection" in {
    a[UnsupportedOperationException] should be thrownBy {
      val splitData = DataSplitUtility.split(
        AutomationUnitTestsUtil.getAdultDf(),
        1,
        "random",
        "label",
        "dbfs:/test",
        "cache",
        "RandomForest",
        2,
        0.7,
        "synth",
        "datetime",
        0.02,
        0.6
      )

      new RandomForestTuner(
        AutomationUnitTestsUtil.getAdultDf(),
        splitData,
        "err"
      ).evolveBest()
    }
  }

  it should " return a valid model for a Binary Classification task" in {
    val _mainConfig = ConfigurationGenerator.generateMainConfig(
      ConfigurationGenerator.generateDefaultConfig("randomforest", "classifier")
    )
    val data = new DataPrep(
      DiscreteTestDataGenerator.generateBinaryClassificationData(10000)
    ).prepData().data
    val trainSplits = DataSplitUtility.split(
      data,
      1,
      "random",
      _mainConfig.labelCol,
      "dbfs:/test",
      "cache",
      "RandomForest",
      2,
      0.7,
      "synth",
      "datetime",
      0.02,
      0.6
    )

    val randomForestModelsWithResults: RandomForestModelsWithResults =
      new RandomForestTuner(data, trainSplits, "classifier")
        .setFirstGenerationGenePool(5)
        .setLabelCol(_mainConfig.labelCol)
        .setFeaturesCol(_mainConfig.featuresCol)
        .setFieldsToIgnore(_mainConfig.fieldsToIgnoreInVector)
        .setRandomForestNumericBoundaries(_mainConfig.numericBoundaries)
        .setRandomForestStringBoundaries(_mainConfig.stringBoundaries)
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
      randomForestModelsWithResults != null,
      "randomForestModelsWithResults should not have been null"
    )
    assert(
      randomForestModelsWithResults.evalMetrics != null,
      "evalMetrics should not have been null"
    )
    assert(
      randomForestModelsWithResults.model != null,
      "model should not have been null"
    )
    assert(
      randomForestModelsWithResults.modelHyperParams != null,
      "modelHyperParams should not have been null"
    )
  }

  it should " return a valid model for a Multi-class Classification task" in {
    val _mainConfig = ConfigurationGenerator.generateMainConfig(
      ConfigurationGenerator.generateDefaultConfig("randomforest", "classifier")
    )
    val data = new DataPrep(
      DiscreteTestDataGenerator.generateMultiClassClassificationData(10000)
    ).prepData().data
    val trainSplits = DataSplitUtility.split(
      data,
      1,
      "random",
      _mainConfig.labelCol,
      "dbfs:/test",
      "cache",
      "RandomForest",
      2,
      0.7,
      "synth",
      "datetime",
      0.02,
      0.6
    )

    val randomForestModelsWithResults: RandomForestModelsWithResults =
      new RandomForestTuner(data, trainSplits, "classifier")
        .setFirstGenerationGenePool(5)
        .setLabelCol(_mainConfig.labelCol)
        .setFeaturesCol(_mainConfig.featuresCol)
        .setFieldsToIgnore(_mainConfig.fieldsToIgnoreInVector)
        .setRandomForestNumericBoundaries(_mainConfig.numericBoundaries)
        .setRandomForestStringBoundaries(_mainConfig.stringBoundaries)
        .setScoringMetric("accuracy")
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
      randomForestModelsWithResults != null,
      "randomForestModelsWithResults should not have been null"
    )
    assert(
      randomForestModelsWithResults.evalMetrics != null,
      "evalMetrics should not have been null"
    )
    assert(
      randomForestModelsWithResults.model != null,
      "model should not have been null"
    )
    assert(
      randomForestModelsWithResults.modelHyperParams != null,
      "modelHyperParams should not have been null"
    )
  }

  it should " return a valid model for a Regression task" in {
    val _mainConfig = ConfigurationGenerator.generateMainConfig(
      ConfigurationGenerator.generateDefaultConfig("randomforest", "regressor")
    )
    val data = new DataPrep(
      DiscreteTestDataGenerator.generateRegressionData(10000)
    ).prepData().data
    val trainSplits = DataSplitUtility.split(
      data,
      1,
      "random",
      _mainConfig.labelCol,
      "dbfs:/test",
      "cache",
      "RandomForest",
      2,
      0.7,
      "synth",
      "datetime",
      0.02,
      0.6
    )

    val randomForestModelsWithResults: RandomForestModelsWithResults =
      new RandomForestTuner(data, trainSplits, "regressor")
        .setFirstGenerationGenePool(5)
        .setLabelCol(_mainConfig.labelCol)
        .setFeaturesCol(_mainConfig.featuresCol)
        .setFieldsToIgnore(_mainConfig.fieldsToIgnoreInVector)
        .setRandomForestNumericBoundaries(
          Map(
            "numTrees" -> (50.0, 100.0),
            "maxBins" -> (30.0, 100.0),
            "maxDepth" -> (2.0, 10.0),
            "minInfoGain" -> (0.3, 0.5),
            "subSamplingRate" -> (0.5, 0.6)
          )
        )
        .setRandomForestStringBoundaries(_mainConfig.stringBoundaries)
        .setScoringMetric("rmse")
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
        .setParallelism(2)
        .setKFold(1)
        .setSeed(_mainConfig.geneticConfig.seed)
        .setOptimizationStrategy("minimize")
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
      randomForestModelsWithResults != null,
      "randomForestModelsWithResults should not have been null"
    )
    assert(
      randomForestModelsWithResults.evalMetrics != null,
      "evalMetrics should not have been null"
    )
    assert(
      randomForestModelsWithResults.model != null,
      "model should not have been null"
    )
    assert(
      randomForestModelsWithResults.modelHyperParams != null,
      "modelHyperParams should not have been null"
    )
  }


}
