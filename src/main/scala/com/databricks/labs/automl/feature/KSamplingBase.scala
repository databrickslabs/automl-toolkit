package com.databricks.labs.automl.feature

import com.databricks.labs.automl.feature.structures.{
  KSamplingConfiguration,
  KSamplingDefaults
}
import com.databricks.labs.automl.utils.SparkSessionWrapper

trait KSamplingBase extends KSamplingDefaults with SparkSessionWrapper {

  final private[feature] val allowableKMeansDistanceMeasurements: List[String] =
    List("cosine", "euclidean")
  final private[feature] val allowableMutationModes: List[String] =
    List("weighted", "random", "ratio")
  final private[feature] val allowableVectorMutationMethods: List[String] =
    List("random", "fixed", "all")

  private[feature] var _featuresCol: String = defaultFeaturesCol
  private[feature] var _labelCol: String = defaultLabelCol
  private[feature] var _syntheticCol: String = defaultSyntheticCol
  private[feature] var _fieldsToIgnore: Array[String] = defaultFieldsToIgnore
  private[feature] var _kGroups: Int = defaultKGroups
  private[feature] var _kMeansMaxIter: Int = defaultKMeansMaxIter
  private[feature] var _kMeansTolerance: Double = defaultKMeansTolerance
  private[feature] var _kMeansDistanceMeasurement: String =
    defaultKMeansDistanceMeasurement
  private[feature] var _kMeansSeed: Long = defaultKMeansSeed
  private[feature] var _kMeansPredictionCol: String = defaultKMeansPredictionCol
  private[feature] var _lshHashTables = defaultHashTables
  private[feature] var _lshSeed = defaultLSHSeed
  private[feature] var _lshOutputCol = defaultLSHOutputCol
  private[feature] var _quorumCount = defaultQuorumCount
  private[feature] var _minimumVectorCountToMutate =
    defaultMinimumVectorCountToMutate
  private[feature] var _vectorMutationMethod = defaultVectorMutationMethod
  private[feature] var _mutationMode = defaultMutationMode
  private[feature] var _mutationValue = defaultMutationValue

  private[feature] var conf = getKSamplingConfig

  /**
    * Setter for the Feature Column name of the input DataFrame
    * @param value String: name of the feature vector column
    * @return this
    */
  def setFeaturesCol(value: String): this.type = {
    _featuresCol = value; setConfig; this
  }

  /**
    * Setter for the Label Column name of the input DataFrame
    * @param value String: name of the label column
    * @return this
    */
  def setLabelCol(value: String): this.type = {
    _labelCol = value
    setConfig
    this
  }

  /**
    * Setter for the name to be used for the synthetic column flag that is attached to the output dataframe as an
    * indication that the data present is generated and not original.
    * @param value String: name to be used throughout the job to delineate the fact that the data in the row is
    *              generated.
    * @return this
    */
  def setSyntheticCol(value: String): this.type = {
    _syntheticCol = value
    setConfig
    this
  }

  /**
    * Setter to provide a listing of any fields that are intended to be ignored in the generated dataframe
    * @param value Array[String]: field names to ignore in the data generation aspect
    * @return this
    */
  def setFieldsToIgnore(value: Array[String]): this.type = {
    _fieldsToIgnore = value
    setConfig
    this
  }

  /**
    * Setter for specifying the number of K-Groups to generate in the KMeans model
    * @param value Int: number of k groups to generate
    * @return this
    */
  def setKGroups(value: Int): this.type = {
    _kGroups = value
    setConfig
    this
  }

  /**
    * Setter for specifying the maximum number of iterations for the KMeans model to go through to converge
    * @param value Int: Maximum limit on iterations
    * @return this
    */
  def setKMeansMaxIter(value: Int): this.type = {
    _kMeansMaxIter = value
    setConfig
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
    setConfig
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
    setConfig
    this
  }

  /**
    * Setter for a KMeans seed for the clustering algorithm
    * @param value Long: Seed value
    * @return this
    */
  def setKMeansSeed(value: Long): this.type = {
    _kMeansSeed = value
    setConfig
    this
  }

  /**
    * Setter for the internal KMeans column for cluster membership attribution
    * @param value String: column name for internal algorithm column for group membership
    * @return this
    */
  def setKMeansPredictionCol(value: String): this.type = {
    _kMeansPredictionCol = value
    setConfig
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
    setConfig
    this
  }

  /**
    * Setter for a MinHashLSH seed value for the model.
    * @param value Long: a seed value
    * @return this
    */
  def setLSHSeed(value: Long): this.type = {
    _lshSeed = value
    setConfig
    this
  }

  /**
    * Setter for the internal LSH output hash information column
    * @param value String: column name for the internal MinHashLSH Model transformation value
    * @return this
    */
  def setLSHOutputCol(value: String): this.type = {
    _lshOutputCol = value
    setConfig
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
    setConfig
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
    setConfig
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
    setConfig
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
    setConfig
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
    setConfig
    this
  }

  /**
    * Private method for setting the configuration instantiation.
    * @return this
    */
  private def setConfig: this.type = {
    conf = KSamplingConfiguration(
      featuresCol = _featuresCol,
      labelCol = _labelCol,
      syntheticCol = _syntheticCol,
      fieldsToIgnore = _fieldsToIgnore,
      kGroups = _kGroups,
      kMeansMaxIter = _kMeansMaxIter,
      kMeansTolerance = _kMeansTolerance,
      kMeansDistanceMeasurement = _kMeansDistanceMeasurement,
      kMeansSeed = _kMeansSeed,
      kMeansPredictionCol = _kMeansPredictionCol,
      lshHashTables = _lshHashTables,
      lshSeed = _lshSeed,
      lshOutputCol = _lshOutputCol,
      quorumCount = _quorumCount,
      minimumVectorCountToMutate = _minimumVectorCountToMutate,
      vectorMutationMethod = _vectorMutationMethod,
      mutationMode = _mutationMode,
      mutationValue = _mutationValue
    )
    this
  }

  /**
    * Public method for returning the current state of the configuration as a new instance of the KSamplingConfiguration
    * @return the current state of the KSamplingConfiguration conf
    */
  def getKSamplingConfig: KSamplingConfiguration = {
    KSamplingConfiguration(
      featuresCol = _featuresCol,
      labelCol = _labelCol,
      syntheticCol = _syntheticCol,
      fieldsToIgnore = _fieldsToIgnore,
      kGroups = _kGroups,
      kMeansMaxIter = _kMeansMaxIter,
      kMeansTolerance = _kMeansTolerance,
      kMeansDistanceMeasurement = _kMeansDistanceMeasurement,
      kMeansSeed = _kMeansSeed,
      kMeansPredictionCol = _kMeansPredictionCol,
      lshHashTables = _lshHashTables,
      lshSeed = _lshSeed,
      lshOutputCol = _lshOutputCol,
      quorumCount = _quorumCount,
      minimumVectorCountToMutate = _minimumVectorCountToMutate,
      vectorMutationMethod = _vectorMutationMethod,
      mutationMode = _mutationMode,
      mutationValue = _mutationValue
    )
  }

  /**
    * Static method for generating the fields to drop from the interstitial dataframes during the algorithm's execution.
    * @return
    */
  private[feature] def fieldsToDrop: List[String] =
    List(_kMeansPredictionCol, _lshOutputCol, "distCol")

}
