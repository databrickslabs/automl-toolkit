package com.databricks.labs.automl.model

import com.databricks.labs.automl.executor.DataPrep
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.model.tools.split.DataSplitUtility
import com.databricks.labs.automl.params.LinearRegressionModelsWithResults
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class LinearRegressionTunerTest extends AbstractUnitSpec {

  "LinearRegressionTuner" should "throw NoSuchElementException for passing invalid params" in {
    a[NullPointerException] should be thrownBy {
      val splitData = DataSplitUtility.split(
        AutomationUnitTestsUtil.getAdultDf(),
        1,
        "random",
        "label",
        "dbfs:/test",
        "cache",
        "LinearRegression",
        2,
        0.7,
        "synth",
        "datetime",
        0.02,
        0.6
      )

      new LinearRegressionTuner(null, splitData).evolveBest()
    }
  }

  it should "should throw NoSuchElementException for passing invalid dataset" in {
    a[AssertionError] should be thrownBy {
      val splitData = DataSplitUtility.split(
        AutomationUnitTestsUtil.getAdultDf(),
        1,
        "random",
        "label",
        "dbfs:/test",
        "cache",
        "LinearRegression",
        2,
        0.7,
        "synth",
        "datetime",
        0.02,
        0.6
      )

      new LinearRegressionTuner(
        AutomationUnitTestsUtil.sparkSession.emptyDataFrame,
        splitData
      ).evolveBest()
    }
  }

  it should "should return valid Regression Model" in {

    val _mainConfig = ConfigurationGenerator.generateMainConfig(
      ConfigurationGenerator
        .generateDefaultConfig("linearregression", "regressor")
    )
    val data = new DataPrep(
      AutomationUnitTestsUtil.convertCsvToDf("/AirQualityUCI.csv")
    ).prepData().data
    val splitData = DataSplitUtility.split(
      data,
      1,
      "random",
      _mainConfig.labelCol,
      "dbfs:/test",
      "cache",
      "LinearRegression",
      2,
      0.7,
      "synth",
      "datetime",
      0.02,
      0.6
    )

    val linearRegressionModelsWithResults: LinearRegressionModelsWithResults =
      new LinearRegressionTuner(data, splitData)
        .setFirstGenerationGenePool(5)
        .setLabelCol(_mainConfig.labelCol)
        .setFeaturesCol(_mainConfig.featuresCol)
        .setFieldsToIgnore(_mainConfig.fieldsToIgnoreInVector)
        .setLinearRegressionNumericBoundaries(
          Map(
            "elasticNetParams" -> Tuple2(0.0, 1.0),
            "maxIter" -> Tuple2(10.0, 100.0),
            "regParam" -> Tuple2(0.2, 1.0),
            "tolerance" -> Tuple2(1E-9, 1E-6)
          )
        )
        .setLinearRegressionStringBoundaries(
          Map("loss" -> List("squaredError"))
        )
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
      linearRegressionModelsWithResults != null,
      "linearRegressionModelsWithResults should not have been null"
    )
    assert(
      linearRegressionModelsWithResults.evalMetrics != null,
      "evalMetrics should not have been null"
    )
    assert(
      linearRegressionModelsWithResults.model != null,
      "model should not have been null"
    )
    assert(
      linearRegressionModelsWithResults.modelHyperParams != null,
      "modelHyperParams should not have been null"
    )
  }

}
