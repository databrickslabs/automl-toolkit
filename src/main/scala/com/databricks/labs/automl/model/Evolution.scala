package com.databricks.labs.automl.model

import com.databricks.labs.automl.params.{
  Defaults,
  EvolutionDefaults,
  KSampleConfig,
  RandomForestConfig
}
import com.databricks.labs.automl.utils.{
  DataValidation,
  SeedConverters,
  SparkSessionWrapper
}
import org.apache.spark.ml.evaluation.{
  BinaryClassificationEvaluator,
  MulticlassClassificationEvaluator,
  RegressionEvaluator
}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{count, _}
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe._

trait Evolution
    extends DataValidation
    with EvolutionDefaults
    with SeedConverters
    with SparkSessionWrapper
    with Defaults {

  var _labelCol: String = _defaultLabel
  var _featureCol: String = _defaultFeature
  var _trainPortion: Double = _defaultTrainPortion
  var _trainSplitMethod: String = _defaultTrainSplitMethod
  var _kSampleConfig: KSampleConfig = _defaultKSampleConfig
  var _trainSplitChronologicalColumn: String =
    _defaultTrainSplitChronologicalColumn
  var _trainSplitChronologicalRandomPercentage: Double =
    _defaultTrainSplitChronologicalRandomPercentage
  var _parallelism: Int = _defaultParallelism
  var _kFold: Int = _defaultKFold
  var _seed: Long = _defaultSeed
  var _kFoldIteratorRange: scala.collection.parallel.immutable.ParRange =
    Range(0, _kFold).par

  var _optimizationStrategy: String = _defaultOptimizationStrategy
  var _firstGenerationGenePool: Int = _defaultFirstGenerationGenePool
  var _numberOfMutationGenerations: Int = _defaultNumberOfMutationGenerations
  var _numberOfParentsToRetain: Int = _defaultNumberOfParentsToRetain
  var _numberOfMutationsPerGeneration: Int =
    _defaultNumberOfMutationsPerGeneration
  var _geneticMixing: Double = _defaultGeneticMixing
  var _generationalMutationStrategy: String =
    _defaultGenerationalMutationStrategy
  var _mutationMagnitudeMode: String = _defaultMutationMagnitudeMode
  var _fixedMutationValue: Int = _defaultFixedMutationValue
  var _earlyStoppingScore: Double = _defaultEarlyStoppingScore
  var _earlyStoppingFlag: Boolean = _defaultEarlyStoppingFlag

  var _evolutionStrategy: String = _defaultEvolutionStrategy
  var _geneticMBOCandidateFactor: Int = _defaultGeneticMBOCandidateFactor
  var _geneticMBORegressorType: String = _defaultGeneticMBORegressorType
  var _continuousEvolutionImprovementThreshold: Int =
    _defaultContinuousEvolutionImprovementThreshold
  var _continuousEvolutionMaxIterations: Int =
    _defaultContinuousEvolutionMaxIterations
  var _continuousEvolutionStoppingScore: Double =
    _defaultContinuousEvolutionStoppingScore
  var _continuousEvolutionParallelism: Int =
    _defaultContinuousEvolutionParallelism
  var _continuousEvolutionMutationAggressiveness: Int =
    _defaultContinuousEvolutionMutationAggressiveness
  var _continuousEvolutionGeneticMixing: Double =
    _defaultContinuousEvolutionGeneticMixing
  var _continuousEvolutionRollingImprovementCount: Int =
    _defaultContinuousEvolutionRollingImprovementCount

  var _initialGenerationMode: String = _defaultFirstGenMode
  var _initialGenerationPermutationCount: Int = _defaultFirstGenPermutations
  var _initialGenerationIndexMixingMode: String =
    _defaultFirstGenIndexMixingMode
  var _initialGenerationArraySeed: Long = _defaultFirstGenArraySeed
  var _hyperSpaceModelCount: Int = _defaultHyperSpaceModelCount

  var _modelSeedSet: Boolean = false
  var _modelSeed: Map[String, Any] = Map.empty

  var _dataReduce: Double = _defaultDataReduce

  var _syntheticCol: String = _defaultKSampleConfig.syntheticCol
  var _kGroups: Int = _defaultKSampleConfig.kGroups
  var _kMeansMaxIter: Int = _defaultKSampleConfig.kMeansMaxIter
  var _kMeansTolerance: Double = _defaultKSampleConfig.kMeansTolerance
  var _kMeansDistanceMeasurement: String =
    _defaultKSampleConfig.kMeansDistanceMeasurement
  var _kMeansSeed: Long = _defaultKSampleConfig.kMeansSeed
  var _kMeansPredictionCol: String = _defaultKSampleConfig.kMeansPredictionCol
  var _lshHashTables: Int = _defaultKSampleConfig.lshHashTables
  var _lshSeed: Long = _defaultKSampleConfig.lshSeed
  var _lshOutputCol: String = _defaultKSampleConfig.lshOutputCol
  var _quorumCount: Int = _defaultKSampleConfig.quorumCount
  var _minimumVectorCountToMutate: Int =
    _defaultKSampleConfig.minimumVectorCountToMutate
  var _vectorMutationMethod: String = _defaultKSampleConfig.vectorMutationMethod
  var _mutationMode: String = _defaultKSampleConfig.mutationMode
  var _mutationValue: Double = _defaultKSampleConfig.mutationValue
  var _labelBalanceMode: String = _defaultKSampleConfig.labelBalanceMode
  var _cardinalityThreshold: Int = _defaultKSampleConfig.cardinalityThreshold
  var _numericRatio: Double = _defaultKSampleConfig.numericRatio
  var _numericTarget: Int = _defaultKSampleConfig.numericTarget

  var _randomizer: scala.util.Random = scala.util.Random
  _randomizer.setSeed(_seed)

  def setLabelCol(value: String): this.type = {
    _labelCol = value
    this
  }

  def setFeaturesCol(value: String): this.type = {
    _featureCol = value
    this
  }

  def setTrainPortion(value: Double): this.type = {
    require(
      value < 1.0 & value > 0.0,
      "Training portion must be in the range > 0 and < 1"
    )
    _trainPortion = value
    this
  }

  def setTrainSplitMethod(value: String): this.type = {
    require(
      allowableTrainSplitMethod.contains(value),
      s"TrainSplitMethod $value must be one of: ${allowableTrainSplitMethod.mkString(", ")}"
    )
    _trainSplitMethod = value
    this
  }

  /**
    * Setter - for setting the name of the Synthetic column name
    * @param value String: A column name that is uniquely not part of the main DataFrame
    * @since 0.5.1
    * @author Ben Wilson
    */
  def setSyntheticCol(value: String): this.type = {
    _syntheticCol = value
    this
  }

  /**
    * Setter for specifying the number of K-Groups to generate in the KMeans model
    * @param value Int: number of k groups to generate
    * @return this
    */
  def setKGroups(value: Int): this.type = {
    _kGroups = value
    this
  }

  /**
    * Setter for specifying the maximum number of iterations for the KMeans model to go through to converge
    * @param value Int: Maximum limit on iterations
    * @return this
    */
  def setKMeansMaxIter(value: Int): this.type = {
    _kMeansMaxIter = value
    this
  }

  /**
    * Setter for Setting the tolerance for KMeans (must be >0)
    * @param value The tolerance value setting for KMeans
    * @see reference: [[http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.clustering.KMeans]]
    *      for further details.
    * @return this
    * @throws IllegalArgumentException() if a value less than 0 is entered
    */
  @throws(classOf[IllegalArgumentException])
  def setKMeansTolerance(value: Double): this.type = {
    require(
      value > 0,
      s"KMeans tolerance value ${value.toString} is out of range.  Must be > 0."
    )
    _kMeansTolerance = value
    this
  }

  /**
    * Setter for which distance measurement to use to calculate the nearness of vectors to a centroid
    * @param value String: Options -> "euclidean" or "cosine" Default: "euclidean"
    * @return this
    * @throws IllegalArgumentException() if an invalid value is entered
    */
  @throws(classOf[IllegalArgumentException])
  def setKMeansDistanceMeasurement(value: String): this.type = {
    require(
      allowableKMeansDistanceMeasurements.contains(value),
      s"Kmeans Distance Measurement $value is not " +
        s"a valid mode of operation.  Must be one of: ${allowableKMeansDistanceMeasurements.mkString(", ")}"
    )
    _kMeansDistanceMeasurement = value
    this
  }

  /**
    * Setter for a KMeans seed for the clustering algorithm
    * @param value Long: Seed value
    * @return this
    */
  def setKMeansSeed(value: Long): this.type = {
    _kMeansSeed = value
    this
  }

  /**
    * Setter for the internal KMeans column for cluster membership attribution
    * @param value String: column name for internal algorithm column for group membership
    * @return this
    */
  def setKMeansPredictionCol(value: String): this.type = {
    _kMeansPredictionCol = value
    this
  }

  /**
    * Setter for Configuring the number of Hash Tables to use for MinHashLSH
    * @param value Int: Count of hash tables to use
    * @see [[http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.MinHashLSH]]
    *     for more information
    * @return this
    */
  def setLSHHashTables(value: Int): this.type = {
    _lshHashTables = value
    this
  }

  /**
    * Setter for the LSH Seed for the model
    * @param value Long: Seed value
    * @return this
    */
  def setLSHSeed(value: Long): this.type = {
    _lshSeed = value
    this
  }

  /**
    * Setter for the internal LSH output hash information column
    * @param value String: column name for the internal MinHashLSH Model transformation value
    * @return this
    */
  def setLSHOutputCol(value: String): this.type = {
    _lshOutputCol = value
    this
  }

  /**
    * Setter for how many vectors to find in adjacency to the centroid for generation of synthetic data
    * @note the higher the value set here, the higher the variance in synthetic data generation
    * @param value Int: Number of vectors to find nearest each centroid within the class
    * @return this
    */
  def setQuorumCount(value: Int): this.type = {
    _quorumCount = value
    this
  }

  /**
    * Setter for minimum threshold for vector indexes to mutate within the feature vector.
    * @note In vectorMutationMethod "fixed" this sets the fixed count of how many vector positions to mutate.
    *       In vectorMutationMethod "random" this sets the lower threshold for 'at least this many indexes will
    *       be mutated'
    * @param value The minimum (or fixed) number of indexes to mutate.
    * @return this
    */
  def setMinimumVectorCountToMutate(value: Int): this.type = {
    _minimumVectorCountToMutate = value
    this
  }

  /**
    * Setter for the Vector Mutation Method
    * @note Options:
    *       "fixed" - will use the value of minimumVectorCountToMutate to select random indexes of this number of indexes.
    *       "random" - will use this number as a lower bound on a random selection of indexes between this and the vector length.
    *       "all" - will mutate all of the vectors.
    * @param value String - the mode to use.
    * @return this
    * @throws IllegalArgumentException() if the mode is not supported.
    */
  @throws(classOf[IllegalArgumentException])
  def setVectorMutationMethod(value: String): this.type = {
    require(
      allowableVectorMutationMethods.contains(value),
      s"Vector Mutation Mode $value is not supported.  " +
        s"Must be one of: ${allowableVectorMutationMethods.mkString(", ")} "
    )
    _vectorMutationMethod = value
    this
  }

  /**
    * Setter for the Mutation Mode of the feature vector individual values
    * @note Options:
    *       "weighted" - uses weighted averaging to scale the euclidean distance between the centroid vector and mutation candidate vectors
    *       "random" - randomly selects a position on the euclidean vector between the centroid vector and the candidate mutation vectors
    *       "ratio" - uses a ratio between the values of the centroid vector and the mutation vector    *
    * @param value String: the mode to use.
    * @return this
    * @throws IllegalArgumentException() if the mode is not supported.
    */
  @throws(classOf[IllegalArgumentException])
  def setMutationMode(value: String): this.type = {
    require(
      allowableMutationModes.contains(value),
      s"Mutation Mode $value is not a valid mode of operation.  " +
        s"Must be one of: ${allowableMutationModes.mkString(", ")}"
    )
    _mutationMode = value
    this
  }

  /**
    * Setter for specifying the mutation magnitude for the modes 'weighted' and 'ratio' in mutationMode
    * @param value Double: value between 0 and 1 for mutation magnitude adjustment.
    * @note the higher this value, the closer to the centroid vector vs. the candidate mutation vector the synthetic row data will be.
    * @return this
    * @throws IllegalArgumentException() if the value specified is outside of the range (0, 1)
    */
  @throws(classOf[IllegalArgumentException])
  def setMutationValue(value: Double): this.type = {
    require(
      value > 0 & value < 1,
      s"Mutation Value must be between 0 and 1. Value $value is not permitted."
    )
    _mutationValue = value
    this
  }

  /**
    * Setter - for determining the label balance approach mode.
    * @note Available modes: <br>
    *         <i>'match'</i>: Will match all smaller class counts to largest class count.  [WARNING] - May significantly increase memory pressure!<br>
    *         <i>'percentage'</i> Will adjust smaller classes to a percentage value of the largest class count.
    *         <i>'target'</i> Will increase smaller class counts to a fixed numeric target of rows.
    * @param value String: one of: 'match', 'percentage' or 'target'
    * @note Default: "percentage"
    * @since 0.5.1
    * @author Ben Wilson
    * @throws UnsupportedOperationException() if the provided mode is not supported.
    */
  @throws(classOf[UnsupportedOperationException])
  def setLabelBalanceMode(value: String): this.type = {
    require(
      allowableLabelBalanceModes.contains(value),
      s"Label Balance Mode $value is not supported." +
        s"Must be one of: ${allowableLabelBalanceModes.mkString(", ")}"
    )
    _labelBalanceMode = value
    this
  }

  /**
    * Setter - for overriding the cardinality threshold exception threshold.  [WARNING] increasing this value on
    * a sufficiently large data set could incur, during runtime, excessive memory and cpu pressure on the cluster.
    * @param value Int: the limit above which an exception will be thrown for a classification problem wherein the
    *              label distinct count is too large to successfully generate synthetic data.
    * @note Default: 20
    * @since 0.5.1
    * @author Ben Wilson
    */
  def setCardinalityThreshold(value: Int): this.type = {
    _cardinalityThreshold = value
    this
  }

  /**
    * Setter - for specifying the percentage ratio for the mode 'percentage' in setLabelBalanceMode()
    * @param value Double: A fractional double in the range of 0.0 to 1.0.
    * @note Setting this value to 1.0 is equivalent to setting the label balance mode to 'match'
    * @note Default: 0.2
    * @since 0.5.1
    * @author Ben Wilson
    * @throws UnsupportedOperationException() if the provided value is outside of the range of 0.0 -> 1.0
    */
  @throws(classOf[UnsupportedOperationException])
  def setNumericRatio(value: Double): this.type = {
    require(
      value <= 1.0 & value > 0.0,
      s"Invalid Numeric Ratio entered!  Must be between 0 and 1." +
        s"${value.toString} is not valid."
    )
    _numericRatio = value
    this
  }

  /**
    * Setter - for specifying the target row count to generate for 'target' mode in setLabelBalanceMode()
    * @param value Int: The desired final number of rows per minority class label
    * @note [WARNING] Setting this value to too high of a number will greatly increase runtime and memory pressure.
    * @since 0.5.1
    * @author Ben Wilson
    */
  def setNumericTarget(value: Int): this.type = {
    _numericTarget = value
    this
  }

  def setTrainSplitChronologicalColumn(value: String): this.type = {
    _trainSplitChronologicalColumn = value
    this
  }

  def setTrainSplitChronologicalRandomPercentage(value: Double): this.type = {
    _trainSplitChronologicalRandomPercentage = value
    if (value > 10)
      println(
        "[WARNING] setTrainSplitChronologicalRandomPercentage() setting this value above 10 " +
          "percent will cause significant per-run train/test skew and variability in row counts during training.  " +
          "Use higher values only if this is desired."
      )
    this
  }

  def setParallelism(value: Int): this.type = {
    require(
      _parallelism < 10000,
      s"Parallelism above 10000 will result in cluster instability."
    )
    _parallelism = value
    this
  }

  def setKFold(value: Int): this.type = {
    _kFold = value
    _kFoldIteratorRange = Range(0, _kFold).par
    this
  }

  def setSeed(value: Long): this.type = {
    _seed = value
    this
  }

  def setOptimizationStrategy(value: String): this.type = {
    val valueLC = value.toLowerCase
    require(
      allowableOptimizationStrategies.contains(valueLC),
      s"Optimization Strategy '$valueLC' is not a member of ${invalidateSelection(valueLC, allowableOptimizationStrategies)}"
    )
    _optimizationStrategy = valueLC
    this
  }

  def setFirstGenerationGenePool(value: Int): this.type = {
    require(
      value >= 5,
      s"Values less than 5 for firstGenerationGenePool will require excessive generational mutation to converge"
    )
    _firstGenerationGenePool = value
    this
  }

  def setNumberOfMutationGenerations(value: Int): this.type = {
    require(value > 0, s"Number of Generations must be greater than 0")
    _numberOfMutationGenerations = value
    this
  }

  def setNumberOfParentsToRetain(value: Int): this.type = {
    require(
      value > 0,
      s"Number of Parents must be greater than 0. '$value' is not a valid number."
    )
    _numberOfParentsToRetain = value
    this
  }

  def setNumberOfMutationsPerGeneration(value: Int): this.type = {
    require(
      value > 0,
      s"Number of Mutations per generation must be greater than 0. '$value' is not a valid number."
    )
    _numberOfMutationsPerGeneration = value
    this
  }

  def setGeneticMixing(value: Double): this.type = {
    require(
      value < 1.0 & value > 0.0,
      s"Mutation Aggressiveness must be in range (0,1). Current Setting of $value is not permitted."
    )
    _geneticMixing = value
    this
  }

  def setGenerationalMutationStrategy(value: String): this.type = {
    val valueLC = value.toLowerCase
    require(
      allowableMutationStrategies.contains(valueLC),
      s"Generational Mutation Strategy '$valueLC' is not a member of ${invalidateSelection(valueLC, allowableMutationStrategies)}"
    )
    _generationalMutationStrategy = valueLC
    this
  }

  def setMutationMagnitudeMode(value: String): this.type = {
    val valueLC = value.toLowerCase
    require(
      allowableMutationMagnitudeMode.contains(valueLC),
      s"Mutation Magnitude Mode '$valueLC' is not a member of ${invalidateSelection(valueLC, allowableMutationMagnitudeMode)}"
    )
    _mutationMagnitudeMode = valueLC
    this
  }

  def setFixedMutationValue(value: Int): this.type = {
    val maxMutationCount = modelConfigLength[RandomForestConfig]
    require(
      value <= maxMutationCount,
      s"Mutation count '$value' cannot exceed number of hyperparameters ($maxMutationCount)"
    )
    require(value > 0, s"Mutation count '$value' must be greater than 0")
    _fixedMutationValue = value
    this
  }

  def setEarlyStoppingScore(value: Double): this.type = {
    _earlyStoppingScore = value
    this
  }

  def setEarlyStoppingFlag(value: Boolean): this.type = {
    _earlyStoppingFlag = value
    this
  }

  def setEvolutionStrategy(value: String): this.type = {
    require(
      allowableEvolutionStrategies.contains(value),
      s"Evolution Strategy '$value' is not a supported mode.  Must be one of: ${invalidateSelection(value, allowableEvolutionStrategies)}"
    )
    _evolutionStrategy = value
    this
  }

  /**
    * Setter for defining the secondary stopping criteria for continuous training mode ( number of consistentlt
    * not-improving runs to terminate the learning algorithm due to diminishing returns.
    * @param value Negative Integer (an improvement to a priori will reset the counter and subsequent non-improvements
    *              will decrement a mutable counter.  If the counter hits this limit specified in value, the continuous
    *              mode algorithm will stop).
    * @author Ben Wilson, Databricks
    * @since 0.6.0
    * @throws IllegalArgumentException if the value is positive.
    */
  @throws(classOf[IllegalArgumentException])
  def setContinuousEvolutionImprovementThreshold(value: Int): this.type = {
    require(
      value < 0,
      s"ContinuousEvolutionImprovementThreshold must be less than zero.  It is " +
        s"recommended to set this value to less than -4."
    )
    _continuousEvolutionImprovementThreshold = value
    this
  }

  /**
    * Setter for selecting the type of Regressor to use for the within-epoch generation MBO of candidates
    * @param value String - one of "XGBoost", "LinearRegression" or "RandomForest"
    * @author Ben Wilson, Databricks
    * @since 0.6.0
    * @throws IllegalArgumentException if the value is not supported
    */
  @throws(classOf[IllegalArgumentException])
  def setGeneticMBORegressorType(value: String): this.type = {
    require(
      allowableMBORegressorTypes.contains(value),
      s"GeneticRegressorType $value is not a supported Regressor " +
        s"Type.  Must be one of: ${allowableMBORegressorTypes.mkString(", ")}"
    )
    _geneticMBORegressorType = value
    this
  }

  /**
    * Setter for defining the factor to be applied to the candidate listing of hyperparameters to generate through
    * mutation for each generation other than the initial and post-modeling optimization phases.  The larger this
    * value (default: 10), the more potential space can be searched.  There is not a large performance hit to this,
    * and as such, values in excess of 100 are viable.
    * @param value Int - a factor to multiply the numberOfMutationsPerGeneration by to generate a count of potential
    *              candidates.
    * @author Ben Wilson, Databricks
    * @since 0.6.0
    * @throws IllegalArgumentException if the value is not greater than zero.
    */
  @throws(classOf[IllegalArgumentException])
  def setGeneticMBOCandidateFactor(value: Int): this.type = {
    require(value > 0, s"GeneticMBOCandidateFactor must be greater than zero.")
    _geneticMBOCandidateFactor = value
    this
  }

  def setContinuousEvolutionMaxIterations(value: Int): this.type = {
    if (value > 500)
      println(
        s"[WARNING] Total Modeling count $value is higher than recommended limit of 500.  " +
          s"This tuning will take a long time to run."
      )
    _continuousEvolutionMaxIterations = value
    this
  }

  def setContinuousEvolutionStoppingScore(value: Double): this.type = {
    _continuousEvolutionStoppingScore = value
    this
  }

  def setContinuousEvolutionParallelism(value: Int): this.type = {
    if (value > 10)
      println(
        s"[WARNING] ContinuousEvolutionParallelism -> $value is higher than recommended " +
          s"concurrency for efficient optimization for convergence." +
          s"\n  Setting this value below 11 will converge faster in most cases."
      )
    _continuousEvolutionParallelism = value
    this
  }

  def setContinuousEvolutionMutationAggressiveness(value: Int): this.type = {
    if (value > 4)
      println(
        s"[WARNING] ContinuousEvolutionMutationAggressiveness -> $value. " +
          s"\n  Setting this higher than 4 will result in extensive random search and will take longer to converge " +
          s"to optimal hyperparameters."
      )
    _continuousEvolutionMutationAggressiveness = value
    this
  }

  def setContinuousEvolutionGeneticMixing(value: Double): this.type = {
    require(
      value < 1.0 & value > 0.0,
      s"Mutation Aggressiveness must be in range (0,1). Current Setting of $value is not permitted."
    )
    _continuousEvolutionGeneticMixing = value
    this
  }

  def setContinuousEvolutionRollingImporvementCount(value: Int): this.type = {
    require(
      value > 0,
      s"ContinuousEvolutionRollingImprovementCount must be > 0. $value is invalid."
    )
    if (value < 10)
      println(
        s"[WARNING] ContinuousEvolutionRollingImprovementCount -> $value setting is low.  " +
          s"Optimal Convergence may not occur due to early stopping."
      )
    _continuousEvolutionRollingImprovementCount = value
    this
  }

  def setModelSeed(value: Map[String, Any]): this.type = {
    _modelSeed = value
    _modelSeedSet = true
    this
  }

  def setDataReductionFactor(value: Double): this.type = {
    require(value > 0, s"Data Reduction Factor must be between 0 and 1")
    require(value < 1, s"Data Reduction Factor must be between 0 and 1")
    _dataReduce = value
    this
  }

  def setFirstGenMode(value: String): this.type = {
    require(
      allowableInitialGenerationModes.contains(value),
      s"First Generation Mode '$value' is not a supported mode." +
        s"  Must be one of: ${invalidateSelection(value, allowableInitialGenerationModes)}"
    )
    _initialGenerationMode = value
    this
  }

  def setFirstGenPermutations(value: Int): this.type = {
    _initialGenerationPermutationCount = value
    this
  }

  def setHyperSpaceModelCount(value: Int): this.type = {
    _hyperSpaceModelCount = value
    this
  }

  def setFirstGenIndexMixingMode(value: String): this.type = {
    require(
      allowableInitialGenerationIndexMixingModes.contains(value),
      s"First Generation Mode '$value' is not a" +
        s"supported mode.  Must be one of ${invalidateSelection(value, allowableInitialGenerationIndexMixingModes)}"
    )
    _initialGenerationIndexMixingMode = value
    this
  }

  def setFirstGenArraySeed(value: Long): this.type = {
    _initialGenerationArraySeed = value
    this
  }

  def getFirstGenArraySeed: Long = _initialGenerationArraySeed

  def getFirstGenIndexMixingMode: String = _initialGenerationIndexMixingMode

  def getFirstGenPermutations: Int = _initialGenerationPermutationCount

  def getFirstGenMode: String = _initialGenerationMode

  def getHyperSpaceModelCount: Int = _hyperSpaceModelCount

  def getLabelCol: String = _labelCol

  def getFeaturesCol: String = _featureCol

  def getTrainPortion: Double = _trainPortion

  def getTrainSplitMethod: String = _trainSplitMethod

  def getTrainSplitChronologicalColumn: String = _trainSplitChronologicalColumn

  def getTrainSplitChronologicalRandomPercentage: Double =
    _trainSplitChronologicalRandomPercentage

  def getParallelism: Int = _parallelism

  def getKFold: Int = _kFold

  def getSeed: Long = _seed

  def getOptimizationStrategy: String = _optimizationStrategy

  def getFirstGenerationGenePool: Int = _firstGenerationGenePool

  def getNumberOfMutationGenerations: Int = _numberOfMutationGenerations

  def getNumberOfParentsToRetain: Int = _numberOfParentsToRetain

  def getNumberOfMutationsPerGeneration: Int = _numberOfMutationsPerGeneration

  def getGeneticMixing: Double = _geneticMixing

  def getGenerationalMutationStrategy: String = _generationalMutationStrategy

  def getMutationMagnitudeMode: String = _mutationMagnitudeMode

  def getFixedMutationValue: Int = _fixedMutationValue

  def getEarlyStoppingScore: Double = _earlyStoppingScore

  def getEarlyStoppingFlag: Boolean = _earlyStoppingFlag

  def getEvolutionStrategy: String = _evolutionStrategy

  def getGeneticMBORegressorType: String = _geneticMBORegressorType

  def getGeneticMBOCandidateFactor: Int = _geneticMBOCandidateFactor

  def getContinuousEvolutionImprovementThreshold: Int =
    _continuousEvolutionImprovementThreshold

  def getContinuousEvolutionMaxIterations: Int =
    _continuousEvolutionMaxIterations

  def getContinuousEvolutionStoppingScore: Double =
    _continuousEvolutionStoppingScore

  def getContinuousEvolutionParallelism: Int = _continuousEvolutionParallelism

  def getContinuousEvolutionMutationAggressiveness: Int =
    _continuousEvolutionMutationAggressiveness

  def getContinuousEvolutionGeneticMixing: Double =
    _continuousEvolutionGeneticMixing

  def getContinuousEvolutionRollingImporvementCount: Int =
    _continuousEvolutionRollingImprovementCount

  def getModelSeed: Map[String, Any] = _modelSeed

  def getDataReductionFactor: Double = _dataReduce

  /**
    * Internal method for validating if a numeric mapping that is specified contains any invalid keys
    * @param standardConfig The static defined numeric mapping for a model type
    * @param modConfig a user-specified mapping override
    * @since 0.6.1
    * @author Ben Wilson, Databricks
    * @throws IllegalArgumentException if the key is invalid for the model type specified.
    */
  @throws(classOf[IllegalArgumentException])
  protected[model] def validateNumericMapping(
    standardConfig: Map[String, (Double, Double)],
    modConfig: Map[String, (Double, Double)]
  ): Unit = {

    val staticKeys = standardConfig.keys.toArray
    val modKeys = modConfig.keys.toArray

    modKeys.foreach(
      x =>
        if (!staticKeys.contains(x))
          throw new IllegalArgumentException(
            s"The numeric Boundary map key " +
              s"supplied: [$x] is not a valid member of Numeric Mapping.  " +
              s"\nKeys are restricted to: [${staticKeys.mkString(", ")}]"
        )
    )

  }

  /**
    * Internal method for validating if a string mapping that is specified contains any invalid keys
    * @param standardConfig The static defined string mapping for a model type
    * @param modConfig a user-specified mapping override
    * @since 0.6.1
    * @author Ben Wilson, Databricks
    * @throws IllegalArgumentException if the key is invalid for the model type specified.
    */
  @throws(classOf[IllegalArgumentException])
  protected[model] def validateStringMapping(
    standardConfig: Map[String, List[String]],
    modConfig: Map[String, List[String]]
  ): Unit = {
    val staticKeys = standardConfig.keys.toArray
    val modKeys = modConfig.keys.toArray

    modKeys.foreach(
      x =>
        if (!staticKeys.contains(x))
          throw new IllegalArgumentException(
            s"The string Boundary map key " +
              s"supplied: [$x] is not a valid member of String Mapping.  " +
              s"\nKeys are restricted to: [${staticKeys.mkString(", ")}]"
        )
    )
  }

  /**
    * Helper function for partially updating a numeric mapping
    * @param defaultMap The default configuration Map for a numeric mapping for model hyperparameter search space
    * @param updateMap user-supplied updated map (doesn't have to have all elements in it)
    * @return The default map, updated with the user-supplied overrides
    * @since 0.6.1
    * @author Ben Wilson, Jas Bali Databricks
    */
  def partialOverrideNumericMapping(
    defaultMap: Map[String, (Double, Double)],
    updateMap: Map[String, (Double, Double)]
  ): Map[String, (Double, Double)] = {

    defaultMap ++ updateMap
  }

  /**
    * Helper function for partially updating a string mapping
    *
    * @param defaultMap The default configuration Map for a string mapping for model hyperparameter search space
    * @param updateMap user-supplied updated map (doesn't have to have all elements in it)
    * @return The default map, updated with the user-supplied overrides
    * @since 0.6.1
    * @author Ben Wilson, Jas Bali Databricks
    */
  def partialOverrideStringMapping(
    defaultMap: Map[String, List[String]],
    updateMap: Map[String, List[String]]
  ): Map[String, List[String]] = {
    defaultMap ++ updateMap
  }

  // TODO - Calculation should take into account early stopping
  def totalModels: Int = _evolutionStrategy match {
    case "batch" =>
      (_numberOfMutationsPerGeneration * _numberOfMutationGenerations) + _firstGenerationGenePool +
        _initialGenerationPermutationCount + _hyperSpaceModelCount
    case "continuous" =>
      _continuousEvolutionMaxIterations - _continuousEvolutionParallelism + _firstGenerationGenePool
    case _ =>
      throw new MatchError(
        s"EvolutionStrategy mode ${_evolutionStrategy} is not supported." +
          s"\n  Choose one of: ${allowableEvolutionStrategies.mkString(", ")}"
      )
  }

  def modelConfigLength[T: TypeTag]: Int = {
    typeOf[T].members
      .collect {
        case m: MethodSymbol if m.isCaseAccessor => m
      }
      .toList
      .length
  }

  private def toDoubleType(x: Any): Option[Double] = x match {
    case i: Int    => Some(i)
    case d: Double => Some(d)
    case _         => None
  }

  private def generateEmptyTrainTest(
    data: DataFrame
  ): (DataFrame, DataFrame) = {

    val schema = data.schema

    var trainData = spark.createDataFrame(sc.emptyRDD[Row], schema)
    var testData = spark.createDataFrame(sc.emptyRDD[Row], schema)
    (trainData, testData)
  }

  /**
    * Method for stratification of the test/train around the unique values of the label column
    * This mode is recommended for label value distributions in classification that have relatively balanced
    * and uniformly distributed instances of the classes.
    * If there is significant skew, it is highly recommended to use under or over sampling.
    *
    * @param data Dataframe that is the input to the train/test split
    * @param seed random seed for splitting the data into train/test.
    * @return An Array of Dataframes: Array[<trainData>, <testData>]
    */
  def stratifiedSplit(data: DataFrame,
                      seed: Long,
                      uniqueLabels: Array[Row]): Array[DataFrame] = {

    var (trainData, testData) = generateEmptyTrainTest(data)

    uniqueLabels.foreach { x =>
      val conversionValue = toDoubleType(x(0)).get

      val Array(trainSplit, testSplit) = data
        .filter(col(_labelCol) === conversionValue)
        .randomSplit(Array(_trainPortion, 1 - _trainPortion), seed)

      trainData = trainData.union(trainSplit)
      testData = testData.union(testSplit)

    }

    Array(trainData, testData)
  }

  def underSampleSplit(data: DataFrame, seed: Long): Array[DataFrame] = {

    var (trainData, testData) = generateEmptyTrainTest(data)

    val totalDataSetCount = data.count()

    val groupedLabelCount = data
      .select(_labelCol)
      .groupBy(_labelCol)
      .agg(count("*").as("counts"))
      .withColumn("skew", col("counts") / lit(totalDataSetCount))
      .select(_labelCol, "skew")

    val uniqueGroups = groupedLabelCount.collect()

    val smallestSkew = groupedLabelCount
      .sort(col("skew").asc)
      .select(col("skew"))
      .first()
      .getDouble(0)

    uniqueGroups.foreach { x =>
      val groupData = toDoubleType(x(0)).get

      val groupRatio = x.getDouble(1)

      val groupDataFrame = data.filter(col(_labelCol) === groupData)

      val Array(train, test) = if (groupRatio == smallestSkew) {
        groupDataFrame.randomSplit(
          Array(_trainPortion, 1 - _trainPortion),
          seed
        )
      } else {
        groupDataFrame
          .sample(false, smallestSkew / groupRatio)
          .randomSplit(Array(_trainPortion, 1 - _trainPortion), seed)
      }

      trainData = trainData.union(train)
      testData = testData.union(test)

    }

    Array(trainData, testData)

  }

  def overSampleSplit(data: DataFrame, seed: Long): Array[DataFrame] = {

    var (trainData, testData) = generateEmptyTrainTest(data)

    val groupedLabelCount = data
      .select(_labelCol)
      .groupBy(_labelCol)
      .agg(count("*").as("counts"))

    val uniqueGroups = groupedLabelCount.collect()

    val largestGroupCount = groupedLabelCount
      .sort(col("counts").desc)
      .select(col("counts"))
      .first()
      .getLong(0)

    uniqueGroups.foreach { x =>
      val groupData = toDoubleType(x(0)).get

      val groupRatio = math.ceil(largestGroupCount / x.getLong(1)).toInt

      for (i <- 1 to groupRatio) {

        val Array(train, test): Array[DataFrame] = data
          .filter(col(_labelCol) === groupData)
          .randomSplit(Array(_trainPortion, 1 - _trainPortion), seed)

        trainData = trainData.union(train)
        testData = testData.union(test)

      }
    }

    Array(trainData, testData)

  }

  def stratifyReduce(data: DataFrame,
                     reductionFactor: Double,
                     seed: Long,
                     uniqueLabels: Array[Row]): Array[DataFrame] = {

    var (trainData, testData) = generateEmptyTrainTest(data)

    uniqueLabels.foreach { x =>
      val conversionValue = toDoubleType(x(0)).get

      val Array(trainSplit, testSplit) = data
        .filter(col(_labelCol) === conversionValue)
        .randomSplit(Array(_trainPortion, 1 - _trainPortion), seed)

      trainData = trainData.union(trainSplit.sample(reductionFactor))
      testData = testData.union(testSplit.sample(reductionFactor))

    }

    Array(trainData, testData)

  }

  def chronologicalSplit(data: DataFrame, seed: Long): Array[DataFrame] = {

    require(
      data.schema.fieldNames.contains(_trainSplitChronologicalColumn),
      s"Chronological Split Field ${_trainSplitChronologicalColumn} is not in schema: " +
        s"${data.schema.fieldNames.mkString(", ")}"
    )

    // Validation check for the random 'wiggle value' if it's set that it won't risk creating zero rows in train set.
    if (_trainSplitChronologicalRandomPercentage > 0.0)
      require(
        (1 - _trainPortion) * _trainSplitChronologicalRandomPercentage / 100 < 0.5,
        s"With trainSplitChronologicalRandomPercentage set at '${_trainSplitChronologicalRandomPercentage}' " +
          s"and a train test ratio of ${_trainPortion} there is a high probability of train data set being empty." +
          s"  \n\tAdjust lower to prevent non-deterministic split levels that could break training."
      )

    // Get the row count
    val rawDataCount = data.count.toDouble

    val splitValue = scala.math.round(rawDataCount * _trainPortion).toInt

    // Get the row number estimation for conduction the split at
    val splitRow: Int = if (_trainSplitChronologicalRandomPercentage <= 0.0) {
      splitValue
    } else {
      // randomly mutate the size of the test validation set
      val splitWiggle = scala.math
        .round(
          rawDataCount * (1 - _trainPortion) *
            _trainSplitChronologicalRandomPercentage / 100
        )
        .toInt
      splitValue - _randomizer.nextInt(splitWiggle)
    }

    // Define the window partition
    val uniqueCol = "chron_grp_autoML_" + java.util.UUID.randomUUID().toString

    // Define temporary non-colliding columns for the window partition
    val uniqueRow = "row_" + java.util.UUID.randomUUID().toString
    val windowDefintion =
      Window.partitionBy(uniqueCol).orderBy(_trainSplitChronologicalColumn)

    // Generate a new Dataframe that has the row number partition, sorted by the chronological field
    val preSplitData = data
      .withColumn(uniqueCol, lit("grp"))
      .withColumn(uniqueRow, row_number() over windowDefintion)
      .drop(uniqueCol)

    // Generate the test/train split data based on sorted chronological column
    Array(
      preSplitData.filter(col(uniqueRow) <= splitRow).drop(uniqueRow),
      preSplitData.filter(col(uniqueRow) > splitRow).drop(uniqueRow)
    )

  }

  /**
    * Split methodology for getting test and train of KSample up-sampled data.<br>
    *   Both data sets are split into test and train. <br>
    *     The returned collections are a union of the real train + synthetic train, but only the real test data.
    * @param data DataFrame: The full data set (containing a synthetic column that indicates whether the data is real or not)
    * @param seed Long: A seed value that is consistent across both data sets
    * @param uniqueLabels Array[Row]: The unique entries of the label values
    * @return Array[DataFrame] of Array(trainData, testData)
    * @since 0.5.1
    * @author Ben Wilson
    */
  def kSamplingSplit(data: DataFrame,
                     seed: Long,
                     uniqueLabels: Array[Row]): Array[DataFrame] = {

    // Split out the real from the synthetic data
    val realData = data.filter(!col(_syntheticCol))

    // Split out the synthetic data
    val syntheticData = data.filter(col(_syntheticCol))

    // Perform stratified splits on both the real and synthetic data
    val Array(realTrain, realTest) =
      stratifiedSplit(realData, seed, uniqueLabels)

    val Array(syntheticTrain, syntheticTest) =
      stratifiedSplit(syntheticData, seed, uniqueLabels)

    // Union the real train data with the synthetic train data and return that with only the real test data
    Array(realTrain.union(syntheticTrain), realTest)

  }

  def genTestTrain(data: DataFrame,
                   seed: Long,
                   uniqueLables: Array[Row]): Array[DataFrame] = {

    _trainSplitMethod match {
      case "random" =>
        data.randomSplit(Array(_trainPortion, 1 - _trainPortion), seed)
      case "chronological" => chronologicalSplit(data, seed)
      case "stratified"    => stratifiedSplit(data, seed, uniqueLables)
      case "overSample"    => overSampleSplit(data, seed)
      case "underSample"   => underSampleSplit(data, seed)
      case "stratifyReduce" =>
        stratifyReduce(data, _dataReduce, seed, uniqueLables)
      case "kSample" => kSamplingSplit(data, seed, uniqueLables)
      case _ =>
        throw new IllegalArgumentException(
          s"Cannot conduct train test split in mode: '${_trainSplitMethod}'"
        )
    }

  }

  def extractBoundaryDouble(
    param: String,
    boundaryMap: Map[String, (AnyVal, AnyVal)]
  ): (Double, Double) = {
    val minimum = boundaryMap(param)._1.asInstanceOf[Double]
    val maximum = boundaryMap(param)._2.asInstanceOf[Double]
    (minimum, maximum)
  }

  def extractBoundaryInteger(
    param: String,
    boundaryMap: Map[String, (AnyVal, AnyVal)]
  ): (Int, Int) = {
    val minimum = boundaryMap(param)._1.asInstanceOf[Double].toInt
    val maximum = boundaryMap(param)._2.asInstanceOf[Double].toInt
    (minimum, maximum)
  }

  def generateRandomDouble(
    param: String,
    boundaryMap: Map[String, (AnyVal, AnyVal)]
  ): Double = {
    val (minimumValue, maximumValue) = extractBoundaryDouble(param, boundaryMap)
    minimumValue + _randomizer.nextDouble() * (maximumValue - minimumValue)
  }

  def generateRandomInteger(param: String,
                            boundaryMap: Map[String, (AnyVal, AnyVal)]): Int = {
    val (minimumValue, maximumValue) =
      extractBoundaryInteger(param, boundaryMap)
    _randomizer.nextInt(maximumValue - minimumValue) + minimumValue
  }

  def generateRandomString(param: String,
                           boundaryMap: Map[String, List[String]]): String = {
    _randomizer.shuffle(boundaryMap(param)).head
  }

  def coinFlip(): Boolean = {
    math.random < 0.5
  }

  def coinFlip(parent: Boolean, child: Boolean, p: Double): Boolean = {
    if (math.random < p) parent else child
  }

  def buildLayerArray(inputFeatureSize: Int,
                      distinctClasses: Int,
                      nLayers: Int,
                      hiddenLayerSizeAdjust: Int): Array[Int] = {

    val layerConstruct = new ArrayBuffer[Int]

    layerConstruct += inputFeatureSize

    (1 to nLayers).foreach { x =>
      layerConstruct += inputFeatureSize + nLayers - x + hiddenLayerSizeAdjust
    }
    layerConstruct += distinctClasses
    layerConstruct.result.toArray
  }

  def generateLayerArray(layerParam: String,
                         layerSizeParam: String,
                         boundaryMap: Map[String, (AnyVal, AnyVal)],
                         inputFeatureSize: Int,
                         distinctClasses: Int): Array[Int] = {

    val layersToGenerate = generateRandomInteger(layerParam, boundaryMap)
    val hiddenLayerSizeAdjust =
      generateRandomInteger(layerSizeParam, boundaryMap)

    buildLayerArray(
      inputFeatureSize,
      distinctClasses,
      layersToGenerate,
      hiddenLayerSizeAdjust
    )

  }

  def getRandomIndeces(minimum: Int,
                       maximum: Int,
                       parameterCount: Int): List[Int] = {
    val fullIndexArray = List.range(0, maximum)
    val randomSeed = new scala.util.Random
    val count = minimum + randomSeed.nextInt((parameterCount - minimum) + 1)
    val adjCount = if (count < 1) 1 else count
    val shuffledArray = scala.util.Random.shuffle(fullIndexArray).take(adjCount)
    shuffledArray.sortWith(_ < _)
  }

  def getFixedIndeces(minimum: Int,
                      maximum: Int,
                      parameterCount: Int): List[Int] = {
    val fullIndexArray = List.range(0, maximum)
    val randomSeed = new scala.util.Random
    randomSeed.shuffle(fullIndexArray).take(parameterCount).sortWith(_ < _)
  }

  def generateMutationIndeces(minimum: Int,
                              maximum: Int,
                              parameterCount: Int,
                              mutationCount: Int): Array[List[Int]] = {
    val mutations = new ArrayBuffer[List[Int]]
    for (_ <- 0 to mutationCount) {
      _mutationMagnitudeMode match {
        case "random" =>
          mutations += getRandomIndeces(minimum, maximum, parameterCount)
        case "fixed" =>
          mutations += getFixedIndeces(minimum, maximum, parameterCount)
        case _ =>
          new UnsupportedOperationException(
            s"Unsupported mutationMagnitudeMode ${_mutationMagnitudeMode}"
          )
      }
    }
    mutations.result.toArray
  }

  def geneMixing(parent: Double,
                 child: Double,
                 parentMutationPercentage: Double): Double = {
    (parent * parentMutationPercentage) + (child * (1 - parentMutationPercentage))
  }

  def geneMixing(parent: Int,
                 child: Int,
                 parentMutationPercentage: Double): Int = {
    ((parent * parentMutationPercentage) + (child * (1 - parentMutationPercentage))).toInt
  }

  def geneMixing(parent: String, child: String): String = {
    val mixed = new ArrayBuffer[String]
    mixed += parent += child
    scala.util.Random.shuffle(mixed.toList).head
  }

  def geneMixing(parent: Array[Int],
                 child: Array[Int],
                 parentMutationPercentage: Double): Array[Int] = {

    val staticStart = parent.head
    val staticEnd = parent.last

    val parentHiddenLayers = parent.length - 2
    val childHiddenLayers = child.length - 2

    val parentMagnitude = parent(1) - staticStart
    val childMagnidue = child(1) - staticStart

    val hiddenLayerMix = geneMixing(
      parentHiddenLayers,
      childHiddenLayers,
      parentMutationPercentage
    )
    val sizeAdjustMix =
      geneMixing(parentMagnitude, childMagnidue, parentMutationPercentage)

    buildLayerArray(staticStart, staticEnd, hiddenLayerMix, sizeAdjustMix)

  }

  /**
    * Method for calculating the remaining time left on the genetic algorithm training (roughly)
    * @note Due to the asynchronous nature of the algorithm, the times are not exact and are a reflection of time
    *       since the creation of the Futures and when they were initially inserted into the thread pool.
    * @param currentGen The current Generation that the model is running on
    * @param currentModel The index of the current model that is being run.
    * @return A Double representing the total completion percentage of the modeling portion of the run.
    * @since 0.2.1
    * @author Ben Wilson
    */
  def calculateModelingFamilyRemainingTime(currentGen: Int,
                                           currentModel: Int): Double = {

    val modelsComplete = _evolutionStrategy match {
      case "batch" =>
        if (currentGen == 1) {
          currentModel
        } else {
          _firstGenerationGenePool + (_numberOfMutationsPerGeneration * (currentGen - 2) + currentModel)
        }
      case _ => currentGen + _firstGenerationGenePool
    }

    (modelsComplete.toDouble / totalModels.toDouble) * 100

  }

  /**
    * Method for validating the distinct class count for a classification type model (for use in determining which
    * evaluator to employ for scoring and optimization of each model)
    * @param df source Dataframe (prior to splitting for train/test)
    * @return Boolean true for Binary Classification problem, false for multi-class problem
    * @since 0.4.0
    * @author Ben Wilson
    */
  def classificationAdjudicator(df: DataFrame): Boolean = {

    // Calculate the distinct entries of the label value for a classification problem
    val uniqueLabelCounts = df.select(_labelCol).distinct().count()

    if (uniqueLabelCounts <= 2) true else false

  }

  /**
    * Method for restricting the available metrics used or are available for optimizing for classification problems
    * @param binaryValidation boolean check from classificationAdjudicator() method
    * @param metricPayload the hard-coded allowable List[String] of allowable classification metrics
    *                      from com.databricks.labs.automl.params.EvolutionDefaults
    * @return a copy of the the allowable params list with the Binary metrics removed if this is a multiclass problem.
    * @since 0.4.0
    * @author Ben Wilson
    */
  def classificationMetricValidator(
    binaryValidation: Boolean,
    metricPayload: List[String]
  ): List[String] = {

    if (binaryValidation) {
      metricPayload
    } else {
      metricPayload.diff(List("areaUnderROC", "areaUnderPR"))
    }

  }

  /**
    * Method for scoring and evaluating classification models (supporting both multi-class and binary classification
    * problems)
    * @param metricName the metric to be tested against (both for binary and multi-class)
    * @param labelColumn the column name in the data set that is the 'source of truth' to compare against
    * @param data the DataFrame that has been transformed
    * @return the score, as a Double value.
    * @since 0.4.0
    * @author Ben Wilson
    */
  def classificationScoring(metricName: String,
                            labelColumn: String,
                            data: DataFrame): Double = {

    metricName match {
      case "areaUnderPR" | "areaUnderROC" =>
        new BinaryClassificationEvaluator()
          .setLabelCol(labelColumn)
          .setRawPredictionCol("probability")
          .setMetricName(metricName)
          .evaluate(data)
      case _ =>
        new MulticlassClassificationEvaluator()
          .setLabelCol(labelColumn)
          .setPredictionCol("prediction")
          .setMetricName(metricName)
          .evaluate(data)
    }

  }

  /**
    * Method for scoring Regression models.
    * @param metricName The metric desired to be tested
    * @param labelColumn The name of the label column
    * @param data the DataFrame that has been transformed by a model.
    * @return the score for the metric, as a Double value.
    * @since 0.4.0
    * @author Ben Wilson
    */
  def regressionScoring(metricName: String,
                        labelColumn: String,
                        data: DataFrame): Double = {

    new RegressionEvaluator()
      .setLabelCol(labelColumn)
      .setMetricName(metricName)
      .evaluate(data)

  }

  def generateAggressiveness(totalConfigs: Int, currentIteration: Int): Int = {
    val mutationAggressiveness = _generationalMutationStrategy match {
      case "linear" =>
        if (totalConfigs - (currentIteration + 1) < 1) 1
        else
          totalConfigs - (currentIteration + 1)
      case _ => _fixedMutationValue
    }
    mutationAggressiveness
  }

}
