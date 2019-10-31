package com.databricks.labs.automl.model.tools

import java.util.UUID

import com.databricks.labs.automl.params.{
  GBTConfig,
  LightGBMConfig,
  LinearRegressionConfig,
  LogisticRegressionConfig,
  MLPCConfig,
  RandomForestConfig,
  SVMConfig,
  TreesConfig,
  XGBoostConfig
}

class ModelReporting(modelType: String, metrics: List[String]) {

  final val _runStart = System.currentTimeMillis / 1000

  /**
    * Private method for generating the run score string
    * @param scoreBattery The collection of scores for each of the scoring methodologies
    * @return The formatted string for reporting out the model validation scores
    * @since 0.5.1
    * @author Ben Wilson
    */
  private def getRunScores(scoreBattery: Map[String, Double]): String = {

    val builtString = new StringBuilder()

    "\n\t\tScores: \n".flatMap(x => builtString += x)

    metrics.foreach { x =>
      s"\t\t\t[$x] -> [${scoreBattery(x)}]\n".flatMap(y => builtString += y)
    }

    builtString.toString

  }

  /**
    * Private method for generating the parameters as a string for stdout and log4j recording of the run information
    * @param config Any: Config for the model hyper parameter collection
    * @return String formatted for the hyper parameters.
    * @since 0.5.1
    * @author Ben Wilson
    */
  private def getParams(config: Any, formatter: String): String = {

    modelType match {
      case "xgboost" =>
        convertXGBoostConfigToHumanReadable(
          config.asInstanceOf[XGBoostConfig],
          formatter
        )
      case "lightgbm" =>
        convertLightGBMConfigToHumanReadable(
          config.asInstanceOf[LightGBMConfig],
          formatter
        )
      case "trees" =>
        convertTreesConfigToHumanReadable(
          config.asInstanceOf[TreesConfig],
          formatter
        )
      case "gbt" =>
        convertGBTConfigToHumanReadable(
          config.asInstanceOf[GBTConfig],
          formatter
        )
      case "linearRegression" =>
        convertLinearRegressionConfigToHumanReadable(
          config.asInstanceOf[LinearRegressionConfig],
          formatter
        )
      case "logisticRegression" =>
        convertLogisticRegressionConfigToHumanReadable(
          config.asInstanceOf[LogisticRegressionConfig],
          formatter
        )
      case "mlpc" =>
        convertMLPCConfigToHumanReadable(
          config.asInstanceOf[MLPCConfig],
          formatter
        )
      case "randomForest" =>
        convertRFConfigToHumanReadable(
          config.asInstanceOf[RandomForestConfig],
          formatter
        )
      case "svm" =>
        convertSVMConfigToHumanReadable(
          config.asInstanceOf[SVMConfig],
          formatter
        )
    }

  }

  /**
    * Private method for getting the current run progress as a formatted string
    * @param runProgressPercentage Utilizes the method from withing Evolution() trait for calculating the
    *                              estimated % complete that the job has achieved thus far.
    * @return String formatted for % complete of the run
    * @since 0.5.1
    * @author Ben Wilson
    */
  private def getRunProgress(runProgressPercentage: Double): String = {
    s"\t\tCurrent Modeling Progress complete for $modelType: " +
      f"$runProgressPercentage%2.4f%%"
  }

  /**
    * Public method for getting the number of seconds since the modeling family job has started and the current process
    * deltas for generation epoch training
    * @param modelStart The kFold start time of the model generation's individual training session
    * @return String formatted time deltas.
    * @since 0.5.1
    * @author Ben Wilson
    */
  def generateModelTime(modelStart: Long): String = {

    val currentTime = System.currentTimeMillis / 1000
    val deltaTime = currentTime - modelStart
    val batteryDelta = currentTime - _runStart

    s"Completed $modelType in $deltaTime seconds. Generation run time: $batteryDelta seconds."

  }

  /**
    * Public method for creating the run start statement as a string based on the uuid of the model run and the
    * hyper parameter settings that are being used.
    * @param runId UUID representing the unique identifier for the generated model
    * @param config The hyper parameter configuration for the run
    * @return String a Human readable string for stdout and logging in log4j
    * @since 0.5.1
    * @author Ben Wilson
    */
  def generateRunStartStatement(runId: UUID, config: Any): String = {
    s"Starting run $runId with Params: ${getParams(config, " ")}"
  }

  /**
    * Public method for reporting on the general completion status of the run.
    * @param generation The current generation that the algorithm is on
    * @param runProgressPercentage The percentage from Evolution() trait for calculating the current completion
    *                              percentage.
    * @return Human readable string for the generation start message.
    * @since 0.5.1
    * @author Ben Wilson
    */
  def generateGenerationStartStatement(
    generation: Int,
    runProgressPercentage: Double
  ): String = {
    f"Starting Generation $generation \n\t\t Completion Status: $runProgressPercentage%2.4f%%"
  }

  /**
    * Public accessor method for generating the print and logging string payload for modeling results and progress
    * @param runId The uuid for the individual model training run
    * @param scoreBattery The resulting score payload for the model (all of the scoring metrics)
    * @param targetMetric The metric that is being used to adjust model tuning selection
    * @param config The hyper parameter configuration for the individual model run
    * @param progress The calculated progress of the model as a Double.
    * @return String block of text that reports out the run results.
    * @since 0.5.1
    * @author Ben Wilson
    */
  def generateRunScoreStatement(runId: UUID,
                                scoreBattery: Map[String, Double],
                                targetMetric: String,
                                config: Any,
                                progress: Double,
                                modelStartTime: Long): String = {

    val scoreText = getRunScores(scoreBattery)

    val outputText = {
      s"\tFinished run $runId with optimaztion target [$targetMetric] value: ${scoreBattery(targetMetric)} " +
        s"\n\tWith full scoring breakdown of: $scoreText" +
        s"\n\tWith hyper-parameters: ${getParams(config, "\n\t\t\t\t")}" +
        s"\n${getRunProgress(progress)}" +
        s"\n\t\t${generateModelTime(modelStartTime)}"
    }

    outputText
  }

  /**
    * Private method for making stdout and logging of params much more readable, particularly for the array objects
    *
    * @param conf The configuration of the run (hyper parameters)
    * @return A string representation that is readable.
    */
  private def convertXGBoostConfigToHumanReadable(conf: XGBoostConfig,
                                                  formatter: String): String = {
    s"\n\t\t\tConfig: $formatter[alpha] -> [${conf.alpha.toString}]" +
      s"$formatter[eta] -> [${conf.eta.toString}]" +
      s"$formatter[gamma] -> [${conf.gamma.toString}]" +
      s"$formatter[lambda] -> [${conf.lambda.toString}]" +
      s"$formatter[maxBins] -> [${conf.maxBins.toString}]" +
      s"$formatter[maxDepth] -> [${conf.maxDepth.toString}]" +
      s"$formatter[minChildWeight] -> [${conf.minChildWeight.toString}]" +
      s"$formatter[numRound] -> [${conf.numRound.toString}]" +
      s"$formatter[subSample] -> [${conf.subSample.toString}]" +
      s"$formatter[trainTestRatio] -> [${conf.trainTestRatio.toString}]"
  }

  private def convertLightGBMConfigToHumanReadable(
    conf: LightGBMConfig,
    formatter: String
  ): String = {
    s"\n\t\t\tConfig: $formatter[baggingFraction] -> [${conf.baggingFraction.toString}]" +
      s"$formatter[baggingFreq] -> [${conf.baggingFreq.toString}]" +
      s"$formatter[featureFreaction] -> [${conf.featureFraction.toString}]" +
      s"$formatter[learningRate] -> [${conf.learningRate.toString}]" +
      s"$formatter[maxBin] -> [${conf.maxBin.toString}]" +
      s"$formatter[maxDepth] -> [${conf.maxDepth.toString}]" +
      s"$formatter[minSumHessianInLeaf] -> [${conf.minSumHessianInLeaf.toString}]" +
      s"$formatter[numIterations] -> [${conf.numIterations.toString}]" +
      s"$formatter[numLeaves] -> [${conf.numLeaves.toString}]" +
      s"$formatter[boostFromAverage] -> [${conf.boostFromAverage.toString}]" +
      s"$formatter[lambdaL1] -> [${conf.lambdaL1.toString}]" +
      s"$formatter[lambdaL2] -> [${conf.lambdaL2.toString}]" +
      s"$formatter[alpha] -> [${conf.alpha.toString}]" +
      s"$formatter[boostingType] -> [${conf.boostingType.toString}]"
  }

  /**
    * Private method for making stdout and logging of params much more readable, particularly for the array objects
    *
    * @param conf The configuration of the run (hyper parameters)
    * @return A string representation that is readable.
    */
  private def convertTreesConfigToHumanReadable(conf: TreesConfig,
                                                formatter: String): String = {
    s"\n\t\t\tConfig: $formatter[impurity] -> [${conf.impurity}]" +
      s"$formatter[maxBins] -> [${conf.maxBins.toString}]" +
      s"$formatter[maxDepth] -> [${conf.maxDepth.toString}]" +
      s"$formatter[minInfoGain] -> [${conf.minInfoGain.toString}]" +
      s"$formatter[minInstancesPerNode] -> [${conf.minInstancesPerNode.toString}]"
  }

  /**
    * Private method for making stdout and logging of params much more readable, particularly for the array objects
    *
    * @param conf The configuration of the run (hyper parameters)
    * @return A string representation that is readable.
    */
  private def convertGBTConfigToHumanReadable(conf: GBTConfig,
                                              formatter: String): String = {
    s"\n\t\t\tConfig: $formatter[impurity] -> [${conf.impurity}]" +
      s"$formatter[lossType] -> [${conf.lossType}]" +
      s"$formatter[maxBins] -> [${conf.maxBins.toString}]" +
      s"$formatter[maxDepth] -> [${conf.maxDepth.toString}]" +
      s"$formatter[maxIter] -> [${conf.maxIter.toString}]" +
      s"$formatter[minInfoGain] -> [${conf.minInfoGain.toString}]" +
      s"$formatter[minInstancesPerNode] -> [${conf.minInstancesPerNode.toString}]" +
      s"$formatter[stepSize] -> [${conf.stepSize.toString}]"
  }

  /**
    * Private method for making stdout and logging of params much more readable, particularly for the array objects
    *
    * @param conf The configuration of the run (hyper parameters)
    * @return A string representation that is readable.
    */
  private def convertLinearRegressionConfigToHumanReadable(
    conf: LinearRegressionConfig,
    formatter: String
  ): String = {
    s"\n\t\t\tConfig: $formatter[elasticNetParams] -> [${conf.elasticNetParams.toString}]" +
      s"$formatter[fitIntercept] -> [${conf.fitIntercept.toString}]" +
      s"$formatter[loss] -> [${conf.loss}]" +
      s"$formatter[maxIter] -> [${conf.maxIter.toString}]" +
      s"$formatter[regParam] -> [${conf.regParam.toString}]" +
      s"$formatter[standardization] -> [${conf.standardization.toString}]" +
      s"$formatter[tolerance] -> [${conf.tolerance.toString}]"
  }

  /**
    * Private method for making stdout and logging of params much more readable, particularly for the array objects
    *
    * @param conf The configuration of the run (hyper parameters)
    * @return A string representation that is readable.
    */
  private def convertLogisticRegressionConfigToHumanReadable(
    conf: LogisticRegressionConfig,
    formatter: String
  ): String = {
    s"\n\t\t\tConfig: $formatter[elasticNetParams] -> [${conf.elasticNetParams.toString}]" +
      s"$formatter[fitIntercept] -> [${conf.fitIntercept.toString}]" +
      s"$formatter[maxIter] -> [${conf.maxIter.toString}]" +
      s"$formatter[regParam] -> [${conf.regParam.toString}]" +
      s"$formatter[standardization] -> [${conf.standardization.toString}]" +
      s"$formatter[tolerance] -> [${conf.tolerance.toString}]"
  }

  /**
    * Private method for making stdout and logging of params much more readable, particularly for the array objects
    * @param conf The configuration of the run (hyper parameters)
    * @return A string representation that is readable.
    */
  private def convertMLPCConfigToHumanReadable(conf: MLPCConfig,
                                               formatter: String): String = {
    s"\n\t\t\tConfig: $formatter[layers] -> [${conf.layers.mkString(",")}]" +
      s"$formatter[maxIter] -> [${conf.maxIter.toString}] $formatter[solver] -> [${conf.solver}]" +
      s"$formatter[stepSize] -> [${conf.stepSize.toString}]$formatter[tolerance] -> [${conf.tolerance.toString}]"
  }

  /**
    * Private method for making stdout and logging of params much more readable, particularly for the array objects
    *
    * @param conf The configuration of the run (hyper parameters)
    * @return A string representation that is readable.
    */
  private def convertRFConfigToHumanReadable(conf: RandomForestConfig,
                                             formatter: String): String = {
    s"\n\t\t\tConfig: $formatter[featureSubsetStrategy] -> [${conf.featureSubsetStrategy}]" +
      s"$formatter[impurity] -> [${conf.impurity}]$formatter[maxBins] -> [${conf.maxBins.toString}]" +
      s"$formatter[maxDepth] -> [${conf.maxDepth.toString}]$formatter[minInfoGain] -> [${conf.minInfoGain.toString}]" +
      s"$formatter[numTrees] -> [${conf.numTrees.toString}]$formatter[subSamplingRate] -> [${conf.subSamplingRate.toString}]"
  }

  /**
    * Private method for making stdout and logging of params much more readable, particularly for the array objects
    *
    * @param conf The configuration of the run (hyper parameters)
    * @return A string representation that is readable.
    */
  private def convertSVMConfigToHumanReadable(conf: SVMConfig,
                                              formatter: String): String = {
    s"\n\t\t\tConfig: $formatter[fitIntercept] -> [${conf.fitIntercept.toString}]" +
      s"$formatter[maxIter] -> [${conf.maxIter.toString}]" +
      s"$formatter[regParam] -> [${conf.regParam.toString}]" +
      s"$formatter[standardization] -> [${conf.standardization.toString}]" +
      s"$formatter[tolerance] -> [${conf.tolerance.toString}]"
  }

}
