package com.databricks.spark.automatedml

import com.databricks.spark.automatedml.executor.DataPrep
import com.databricks.spark.automatedml.model._
import com.databricks.spark.automatedml.params._
import com.databricks.spark.automatedml.reports.{DecisionTreeSplits, RandomForestFeatureImportance}
import com.databricks.spark.automatedml.tracking.MLFlowTracker
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer

class AutomationRunner(df: DataFrame) extends DataPrep(df) {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private def runRandomForest(startingSeed: Option[RandomForestConfig]=None):
  (Array[RandomForestModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = prepData()

    //  TODO: Enable Config for Storage Level Override
    //  TODO: Implement this for all model types
    val cachedData = data.persist(StorageLevel.MEMORY_AND_DISK)
    cachedData.count

    val (modelResults, modelStats) = new RandomForestTuner(cachedData, modelSelection)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setRandomForestNumericBoundaries(_mainConfig.numericBoundaries)
      .setRandomForestStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(_mainConfig.geneticConfig.trainSplitMethod)
      .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
      .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
      .setParallelism(_mainConfig.geneticConfig.parallelism)
      .setKFold(_mainConfig.geneticConfig.kFold)
      .setSeed(_mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
      .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
      .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
      .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
      .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
      .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
      .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
      .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
      .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
      .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
      .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
      .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
      .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
      .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)
      .evolveWithScoringDF(startingSeed)

    cachedData.unpersist()

    (modelResults, modelStats, modelSelection)
  }

  private def runMLPC(): (Array[MLPCModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = prepData()

    modelSelection match {
      case "classifier" =>
        val (modelResults, modelStats) = new MLPCTuner(data)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setMlpcNumericBoundaries(_mainConfig.numericBoundaries)
          .setMlpcStringBoundaries(_mainConfig.stringBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(_mainConfig.geneticConfig.trainSplitMethod)
          .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
          .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
          .setParallelism(_mainConfig.geneticConfig.parallelism)
          .setKFold(_mainConfig.geneticConfig.kFold)
          .setSeed(_mainConfig.geneticConfig.seed)
          .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
          .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
          .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
          .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
          .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
          .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
          .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
          .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
          .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
          .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
          .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
          .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
          .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
          .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
          .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)
          .evolveWithScoringDF()

        (modelResults, modelStats, modelSelection)
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by MultiLayer Perceptron Classifier")
    }
  }

  private def runGBT(): (Array[GBTModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = prepData()

    val (modelResults, modelStats) = new GBTreesTuner(data, modelSelection)
      .setLabelCol(_mainConfig.labelCol)
      .setFeaturesCol(_mainConfig.featuresCol)
      .setGBTNumericBoundaries(_mainConfig.numericBoundaries)
      .setGBTStringBoundaries(_mainConfig.stringBoundaries)
      .setScoringMetric(_mainConfig.scoringMetric)
      .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
      .setTrainSplitMethod(_mainConfig.geneticConfig.trainSplitMethod)
      .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
      .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
      .setParallelism(_mainConfig.geneticConfig.parallelism)
      .setKFold(_mainConfig.geneticConfig.kFold)
      .setSeed(_mainConfig.geneticConfig.seed)
      .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
      .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
      .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
      .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
      .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
      .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
      .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
      .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
      .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
      .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
      .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
      .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
      .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
      .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
      .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
      .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
      .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
      .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)
      .evolveWithScoringDF()

    (modelResults, modelStats, modelSelection)
  }

  private def runLinearRegression(): (Array[LinearRegressionModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = prepData()

    modelSelection match {
      case "regressor" =>
        val (modelResults, modelStats) = new LinearRegressionTuner(data)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setLinearRegressionNumericBoundaries(_mainConfig.numericBoundaries)
          .setLinearRegressionStringBoundaries(_mainConfig.stringBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(_mainConfig.geneticConfig.trainSplitMethod)
          .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
          .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
          .setParallelism(_mainConfig.geneticConfig.parallelism)
          .setKFold(_mainConfig.geneticConfig.kFold)
          .setSeed(_mainConfig.geneticConfig.seed)
          .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
          .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
          .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
          .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
          .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
          .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
          .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
          .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
          .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
          .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
          .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
          .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
          .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
          .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
          .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)
          .evolveWithScoringDF()

        (modelResults, modelStats, modelSelection)
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by Linear Regression")
    }

  }

  private def runLogisticRegression(): (Array[LogisticRegressionModelsWithResults], DataFrame,  String) = {

    val (data, fields, modelSelection) = prepData()

    modelSelection match {
      case "classifier" =>
        val (modelResults, modelStats) = new LogisticRegressionTuner(data)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setLogisticRegressionNumericBoundaries(_mainConfig.numericBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(_mainConfig.geneticConfig.trainSplitMethod)
          .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
          .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
          .setParallelism(_mainConfig.geneticConfig.parallelism)
          .setKFold(_mainConfig.geneticConfig.kFold)
          .setSeed(_mainConfig.geneticConfig.seed)
          .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
          .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
          .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
          .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
          .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
          .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
          .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
          .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
          .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
          .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
          .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
          .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
          .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
          .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
          .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)
          .evolveWithScoringDF()

        (modelResults, modelStats, modelSelection)
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by Logistic Regression")
    }

  }

  private def runSVM(): (Array[SVMModelsWithResults], DataFrame, String) = {

    val (data, fields, modelSelection) = prepData()

    modelSelection match {
      case "classifier" =>
        val (modelResults, modelStats) = new SVMTuner(data)
          .setLabelCol(_mainConfig.labelCol)
          .setFeaturesCol(_mainConfig.featuresCol)
          .setSvmNumericBoundaries(_mainConfig.numericBoundaries)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setScoringMetric(_mainConfig.scoringMetric)
          .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
          .setTrainSplitMethod(_mainConfig.geneticConfig.trainSplitMethod)
          .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
          .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
          .setParallelism(_mainConfig.geneticConfig.parallelism)
          .setKFold(_mainConfig.geneticConfig.kFold)
          .setSeed(_mainConfig.geneticConfig.seed)
          .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
          .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
          .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
          .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
          .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
          .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
          .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
          .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
          .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
          .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
          .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
          .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
          .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
          .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
          .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
          .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
          .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
          .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
          .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)
          .evolveWithScoringDF()

        (modelResults, modelStats, modelSelection)
      case _ => throw new UnsupportedOperationException(
        s"Detected Model Type $modelSelection is not supported by Support Vector Machines")
    }
  }

 private def runTrees(): (Array[TreesModelsWithResults], DataFrame, String) = {

   val (data, fields, modelSelection) = prepData()

   val (modelResults, modelStats) = new DecisionTreeTuner(data, modelSelection)
     .setLabelCol(_mainConfig.labelCol)
     .setFeaturesCol(_mainConfig.featuresCol)
     .setTreesNumericBoundaries(_mainConfig.numericBoundaries)
     .setTreesStringBoundaries(_mainConfig.stringBoundaries)
     .setScoringMetric(_mainConfig.scoringMetric)
     .setScoringMetric(_mainConfig.scoringMetric)
     .setTrainPortion(_mainConfig.geneticConfig.trainPortion)
     .setTrainSplitMethod(_mainConfig.geneticConfig.trainSplitMethod)
     .setTrainSplitChronologicalColumn(_mainConfig.geneticConfig.trainSplitChronologicalColumn)
     .setTrainSplitChronologicalRandomPercentage(_mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage)
     .setParallelism(_mainConfig.geneticConfig.parallelism)
     .setKFold(_mainConfig.geneticConfig.kFold)
     .setSeed(_mainConfig.geneticConfig.seed)
     .setOptimizationStrategy(_mainConfig.scoringOptimizationStrategy)
     .setFirstGenerationGenePool(_mainConfig.geneticConfig.firstGenerationGenePool)
     .setNumberOfMutationGenerations(_mainConfig.geneticConfig.numberOfGenerations)
     .setNumberOfMutationsPerGeneration(_mainConfig.geneticConfig.numberOfMutationsPerGeneration)
     .setNumberOfParentsToRetain(_mainConfig.geneticConfig.numberOfParentsToRetain)
     .setGeneticMixing(_mainConfig.geneticConfig.geneticMixing)
     .setGenerationalMutationStrategy(_mainConfig.geneticConfig.generationalMutationStrategy)
     .setMutationMagnitudeMode(_mainConfig.geneticConfig.mutationMagnitudeMode)
     .setFixedMutationValue(_mainConfig.geneticConfig.fixedMutationValue)
     .setEarlyStoppingFlag(_mainConfig.autoStoppingFlag)
     .setEarlyStoppingScore(_mainConfig.autoStoppingScore)
     .setEvolutionStrategy(_mainConfig.geneticConfig.evolutionStrategy)
     .setContinuousEvolutionMaxIterations(_mainConfig.geneticConfig.continuousEvolutionMaxIterations)
     .setContinuousEvolutionStoppingScore(_mainConfig.geneticConfig.continuousEvolutionStoppingScore)
     .setContinuousEvolutionParallelism(_mainConfig.geneticConfig.continuousEvolutionParallelism)
     .setContinuousEvolutionMutationAggressiveness(_mainConfig.geneticConfig.continuousEvolutionMutationAggressiveness)
     .setContinuousEvolutionGeneticMixing(_mainConfig.geneticConfig.continuousEvolutionGeneticMixing)
     .setContinuousEvolutionRollingImporvementCount(_mainConfig.geneticConfig.continuousEvolutionRollingImprovementCount)
     .evolveWithScoringDF()

   (modelResults, modelStats, modelSelection)
 }

  private def logResultsToMlFlow(runData: Array[GenericModelReturn], modelFamily: String, modelType: String): String = {

    val mlFlowLogger = new MLFlowTracker()
      .setMlFlowTrackingURI(_mainConfig.mlFlowConfig.mlFlowTrackingURI)
      .setMlFlowHostedAPIToken(_mainConfig.mlFlowConfig.mlFlowAPIToken)
      .setMlFlowExperimentName(_mainConfig.mlFlowConfig.mlFlowExperimentName)
      .setModelSaveDirectory(_mainConfig.mlFlowConfig.mlFlowModelSaveDirectory)

    try {
      mlFlowLogger.logMlFlowDataAndModels(runData, modelFamily, modelType)
      "Logged to MlFlow Successful"
    } catch {
      case e: Exception =>
        val stackTrace : String = e.getStackTrace.mkString("\n")
        println(s"Failed to log to mlflow. Check configuration. \n  $stackTrace")
        logger.log(Level.INFO, stackTrace)
        "Failed to Log to MlFlow"
    }

  }

  def exploreFeatureImportances(): (RandomForestModelsWithResults, DataFrame, Array[String]) = {

    val (data, fields, modelType) = prepData()

    val cachedData = data.persist(StorageLevel.MEMORY_AND_DISK)
    cachedData.count

    val featureResults = new RandomForestFeatureImportance(cachedData, _featureImportancesConfig, modelType)
      .setCutoffType(_mainConfig.featureImportanceCutoffType)
      .setCutoffValue(_mainConfig.featureImportanceCutoffValue)
      .runFeatureImportances(fields)
    cachedData.unpersist()

    featureResults
  }

  def runWithFeatureCulling(startingSeed: Option[Map[String, Any]]=None):
  (Array[GenericModelReturn], Array[GenerationalReport], DataFrame, DataFrame) = {

    // Get the Feature Importances

    val (modelResults, importanceDF, culledFields) = exploreFeatureImportances()

    val selectableFields = culledFields :+ _mainConfig.labelCol

    val dataSubset = df.select(selectableFields.map(col):_*).persist(StorageLevel.MEMORY_AND_DISK)
    dataSubset.count
    
    val runResults = new AutomationRunner(dataSubset).setMainConfig(_mainConfig).runModels(startingSeed)

    dataSubset.unpersist()

    runResults
  }

  def generateDecisionSplits(): (String, DataFrame, Any) = {

    val (data, fields, modelType) = prepData()

    new DecisionTreeSplits(data, _treeSplitsConfig, modelType).runTreeSplitAnalysis(fields)

  }





  // TODO: TEST THIS and then move it to Automation Tools.


  def generateRandomForestConfig(configMap: Map[String, Any]): RandomForestConfig = {
    RandomForestConfig(
      numTrees=configMap("numTrees").asInstanceOf[Int],
      impurity=configMap("impurity").asInstanceOf[String],
      maxBins=configMap("maxBins").asInstanceOf[Int],
      maxDepth=configMap("maxDepth").asInstanceOf[Int],
      minInfoGain=configMap("minInfoGain").asInstanceOf[Double],
      subSamplingRate=configMap("subSamplingRate").asInstanceOf[Double],
      featureSubsetStrategy=configMap("featureSubsetStrategy").asInstanceOf[String]
    )
  }


// TODO: add a setter that allows for a Map to be submitted to 'jump start' a training run with a seed value?

  //TODO: log the Generic Return result for each run to Mlflow as a tagged value string. (tag = g)


  def run(seedString: Option[String]=None):
  (Array[GenericModelReturn], Array[GenerationalReport], DataFrame, DataFrame) = {

    if(seedString.nonEmpty) runModels(Option(extractGenericModelReturnMap(seedString.asInstanceOf[String])))
    else runModels()

  }

  def runModels(startingSeed: Option[Map[String, Any]]=None):
  (Array[GenericModelReturn], Array[GenerationalReport], DataFrame, DataFrame) = {

    val genericResults = new ArrayBuffer[GenericModelReturn]

    val (resultArray, modelStats, modelSelection) = _mainConfig.modelFamily match {
      case "RandomForest" =>
        val (results, stats, selection) = startingSeed match {
          case Some(`startingSeed`) =>
            runRandomForest(Option(generateRandomForestConfig(startingSeed.asInstanceOf[Map[String, Any]])))
          case _ => runRandomForest()
        }
        results.foreach{ x=>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection)
      case "GBT" =>
        val (results, stats, selection) = runGBT()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection)
      case "MLPC" =>
        val (results, stats, selection) = runMLPC()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection)
      case "LinearRegression" =>
        val (results, stats, selection) = runLinearRegression()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection)
      case "LogisticRegression" =>
        val (results, stats, selection) = runLogisticRegression()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection)
      case "SVM" =>
        val (results, stats, selection) = runSVM()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection)
      case "Trees" =>
        val (results, stats, selection) = runTrees()
        results.foreach{x =>
          genericResults += GenericModelReturn(
            hyperParams = extractPayload(x.modelHyperParams),
            model = x.model,
            score = x.score,
            metrics = x.evalMetrics,
            generation = x.generation
          )
        }
        (genericResults, stats, selection)
    }

    val genericResultData = genericResults.result.toArray

    if(_mainConfig.mlFlowLoggingFlag) {
      val mlFlowResult = logResultsToMlFlow(genericResultData, _mainConfig.modelFamily, modelSelection)
      println(mlFlowResult)
      logger.log(Level.INFO, mlFlowResult)
    }

    val generationalData = extractGenerationalScores(genericResultData, _mainConfig.scoringOptimizationStrategy,
      _mainConfig.modelFamily, modelSelection)

  (genericResultData, generationalData, modelStats, generationDataFrameReport(generationalData,
    _mainConfig.scoringOptimizationStrategy))
  }





  //TODO: add a generational runner to find the best model in a modelType (classification / regression)
  //TODO: this will require a new configuration methodology (generationalRunnerConfig) that has all of the families
  //TODO: default configs within it. with setters to override individual parts.  Might want to make it its own class.
}
