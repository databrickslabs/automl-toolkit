package com.databricks.labs.automl.model

import com.databricks.labs.automl.executor.DataPrep
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.model.tools.split.DataSplitUtility
import com.databricks.labs.automl.params.GBTModelsWithResults
import com.databricks.labs.automl.{
  AbstractUnitSpec,
  AutomationUnitTestsUtil,
  DiscreteTestDataGenerator
}

class GBTreesTunerTest extends AbstractUnitSpec {

  "GBTreesTuner" should "throw UnsupportedOperationException for passing invalid params" in {
    a[UnsupportedOperationException] should be thrownBy {
      val splitData = DataSplitUtility.split(
        AutomationUnitTestsUtil.getAdultDf(),
        1,
        "random",
        "label",
        "dbfs:/test",
        "cache",
        "GBT",
        2,
        0.7,
        "synth",
        "datetime",
        0.02,
        0.6
      )

      new GBTreesTuner(null, splitData, null).evolveBest()
    }
  }

  it should "should throw UnsupportedOperationException for passing invalid modelSelection" in {
    a[UnsupportedOperationException] should be thrownBy {
      val splitData = DataSplitUtility.split(
        AutomationUnitTestsUtil.getAdultDf(),
        1,
        "random",
        "label",
        "dbfs:/test",
        "cache",
        "GBT",
        2,
        0.7,
        "synth",
        "datetime",
        0.02,
        0.6
      )

      new GBTreesTuner(AutomationUnitTestsUtil.getAdultDf(), splitData, "err")
        .evolveBest()
    }
  }

  it should "should return valid GBTClassifier for Binary Classification Data" in {

    val _mainConfig = ConfigurationGenerator.generateMainConfig(
      ConfigurationGenerator.generateDefaultConfig("gbt", "classifier")
    )
    val data = new DataPrep(
      DiscreteTestDataGenerator.generateBinaryClassificationData(10000)
    ).prepData().data
    val splitData = DataSplitUtility.split(
      data,
      1,
      "random",
      "label",
      "dbfs:/test",
      "cache",
      "GBT",
      2,
      0.7,
      "synth",
      "datetime",
      0.02,
      0.6
    )
    val gbtModelsWithResults: GBTModelsWithResults =
      new GBTreesTuner(data, splitData, "classifier")
        .setFirstGenerationGenePool(5)
        .setLabelCol(_mainConfig.labelCol)
        .setFeaturesCol(_mainConfig.featuresCol)
        .setFieldsToIgnore(_mainConfig.fieldsToIgnoreInVector)
        .setGBTNumericBoundaries(_mainConfig.numericBoundaries)
        .setGBTStringBoundaries(_mainConfig.stringBoundaries)
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
      gbtModelsWithResults != null,
      "gbtModelsWithResults should not have been null"
    )
    assert(
      gbtModelsWithResults.evalMetrics != null,
      "evalMetrics should not have been null"
    )
    assert(
      gbtModelsWithResults.model != null,
      "model should not have been null"
    )
    assert(
      gbtModelsWithResults.modelHyperParams != null,
      "modelHyperParams should not have been null"
    )
  }

  it should "should throw an exception for attempting to run Multiclass Classification in GBT" in {

    val _mainConfig = ConfigurationGenerator.generateMainConfig(
      ConfigurationGenerator.generateDefaultConfig("gbt", "classifier")
    )
    val data =
      new DataPrep(
        DiscreteTestDataGenerator.generateMultiClassClassificationData(10000)
      ).prepData().data

    val splitData = DataSplitUtility.split(
      data,
      1,
      "random",
      _mainConfig.labelCol,
      "dbfs:/test",
      "cache",
      "GBT",
      2,
      0.7,
      "synth",
      "datetime",
      0.02,
      0.6
    )

    val gbtModelsWithResults: GBTreesTuner =
      new GBTreesTuner(data, splitData, "classifier")
        .setFirstGenerationGenePool(5)
        .setLabelCol(_mainConfig.labelCol)
        .setFeaturesCol(_mainConfig.featuresCol)
        .setFieldsToIgnore(_mainConfig.fieldsToIgnoreInVector)
        .setGBTNumericBoundaries(_mainConfig.numericBoundaries)
        .setGBTStringBoundaries(_mainConfig.stringBoundaries)
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

    intercept[IllegalArgumentException] {
      gbtModelsWithResults.evolveBest()
    }

  }

  it should "should return valid GBTRegressor for Regression Data" in {

    val _mainConfig = ConfigurationGenerator.generateMainConfig(
      ConfigurationGenerator.generateDefaultConfig("gbt", "regressor")
    )
    val data = new DataPrep(
      DiscreteTestDataGenerator.generateRegressionData(10000)
    ).prepData().data
    val splitData = DataSplitUtility.split(
      data,
      1,
      "random",
      _mainConfig.labelCol,
      "dbfs:/test",
      "cache",
      "GBT",
      2,
      0.7,
      "synth",
      "datetime",
      0.02,
      0.6
    )

    val gbtModelsWithResults: GBTModelsWithResults =
      new GBTreesTuner(data, splitData, "regressor")
        .setFirstGenerationGenePool(5)
        .setLabelCol(_mainConfig.labelCol)
        .setFeaturesCol(_mainConfig.featuresCol)
        .setFieldsToIgnore(_mainConfig.fieldsToIgnoreInVector)
        .setGBTNumericBoundaries(_mainConfig.numericBoundaries)
        .setGBTStringBoundaries(_mainConfig.stringBoundaries)
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
        .setParallelism(4)
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
      gbtModelsWithResults != null,
      "gbtModelsWithResults should not have been null"
    )
    assert(
      gbtModelsWithResults.evalMetrics != null,
      "evalMetrics should not have been null"
    )
    assert(
      gbtModelsWithResults.model != null,
      "model should not have been null"
    )
    assert(
      gbtModelsWithResults.modelHyperParams != null,
      "modelHyperParams should not have been null"
    )
  }

}
