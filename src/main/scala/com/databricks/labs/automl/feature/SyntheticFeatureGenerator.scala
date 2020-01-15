package com.databricks.labs.automl.feature

import com.databricks.labs.automl.feature.structures.{
  CardinalityPayload,
  RowGenerationConfig
}
import com.databricks.labs.automl.feature.tools.LabelValidation
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class SyntheticFeatureGenerator(data: DataFrame)
    extends SparkSessionWrapper
    with SyntheticFeatureBase
    with KSamplingBase {

  private var _labelBalanceMode: String = defaultLabelBalanceMode
  private var _cardinalityThreshold: Int = defaultCardinalityThreshold
  private var _numericRatio: Double = defaultNumericRatio
  private var _numericTarget: Int = defaultNumericTarget

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

  /**
    * Private method for detecting the primary class, segregating it, and returning the remaining minority classes
    * in a collection
    * @param full The entire cardinality result for the data set
    * @return
    */
  def getMaxAndRest(
    full: Array[CardinalityPayload]
  ): (CardinalityPayload, Array[CardinalityPayload]) = {

    val sortedValues = full.sortWith(_.labelCounts > _.labelCounts)

    (sortedValues.head, sortedValues.drop(1))
  }

  /**
    * Private method for calculating the targets for all smaller classes for the percentage mode
    * @param max The most frequently occurring label
    * @param rest The remaining labels
    * @return Array[RowGenerationConfig] to supply the candidate target numbers for KSampling
    * @throws RuntimeException if the configuration will not result in any KSampling synthetic rows to be generated.
    * @since 0.5.1
    * @author Ben Wilson
    */
  @throws(classOf[RuntimeException])
  def percentageTargets(
    max: CardinalityPayload,
    rest: Array[CardinalityPayload]
  ): Array[RowGenerationConfig] = {

    val targetValue = max.labelCounts

    if (rest.last.labelCounts > math.floor(targetValue * _numericRatio).toInt)
      throw new RuntimeException(
        s"The ratio target of label counts for the smallest minority class ${rest.last.labelValue} (count: " +
          s"${rest.last.labelCounts}) is already above the target" +
          s"threshold value of ${math.floor(targetValue * _numericRatio).toInt}.  " +
          s"Revisit the configuration settings made in setNumericRatio() for KSampling Configuration."
      )

    rest
      .map { x =>
        val targetCounts = math
          .floor(targetValue * _numericRatio)
          .toInt - x.labelCounts

        if (targetCounts > 0) {

          RowGenerationConfig(x.labelValue, targetCounts)
        } else RowGenerationConfig(x.labelValue, 0)
      }
      .filter(x => x.targetCount > 0)

  }

  /**
    * Private method for generating the row count targets for each minority class label for the target mode
    * @param max The most frequently occurring label
    * @param rest The remaining labels
    * @return Array[RowGenerationConfig] to supply the candidate target numbers for KSampling
    * @throws RuntimeException if the configuration will not result in any KSampling synthetic rows to be generated.
    * @since 0.5.1
    * @author Ben Wilson
    */
  def targetValidation(
    max: CardinalityPayload,
    rest: Array[CardinalityPayload]
  ): Array[RowGenerationConfig] = {

    if (rest.last.labelCounts > _numericTarget)
      throw new RuntimeException(
        s"The target value of label counts ${_numericTarget} for KSampling class label target match" +
          s"for the smallest minority class ${rest.last.labelValue} (count: ${rest.last.labelCounts})is " +
          s"already above the target value.  Revisit the settings made in " +
          s"setNumericTarget(). "
      )

    rest
      .filterNot(x => x.labelCounts > _numericTarget)
      .map { x =>
        RowGenerationConfig(x.labelValue, _numericTarget - x.labelCounts)
      }
  }

  /**
    * Private method for generating the row count target for each minority class label for the match mode
    * @param max The most frequently occurring label
    * @param rest The remaining labels
    * @return Array[RowGenerationConfig] to supply the candidate target numbers for KSampling
    * @since 0.5.1
    * @author Ben Wilson
    */
  def matchValidation(
    max: CardinalityPayload,
    rest: Array[CardinalityPayload]
  ): Array[RowGenerationConfig] = {
    rest.map { x =>
      RowGenerationConfig(x.labelValue, max.labelCounts - x.labelCounts)
    }
  }

  /**
    * Private method for generating the row config objects that KSampling requires for label targets
    * @return Array[RowGeneration] for input to KSampling processing
    * @since 0.5.1
    * @author Ben Wilson
    */
  def determineRatios(): Array[RowGenerationConfig] = {

    val generatedGroups =
      LabelValidation(data, _labelCol, _cardinalityThreshold)

    val (max, rest) = getMaxAndRest(generatedGroups)

    _labelBalanceMode match {
      case "percentage" => percentageTargets(max, rest)
      case "target"     => targetValidation(max, rest)
      case "match"      => matchValidation(max, rest)
    }

  }

  def upSample(): DataFrame = {

    // Get the label statistics
    val labelPayload = determineRatios()

    // Generate synthetic data
    val syntheticData = KSampling(
      data = data,
      labelValues = labelPayload,
      featuresCol = _featuresCol,
      labelsCol = _labelCol,
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

    // Merge the original DataFrame with the synthetic data
    data.withColumn(_syntheticCol, lit(false)).union(syntheticData)

  }

}

object SyntheticFeatureGenerator {
  def apply(data: DataFrame,
            featuresCol: String,
            labelCol: String,
            syntheticCol: String,
            fieldsToIgnore: Array[String],
            kGroups: Int,
            kMeansMaxIter: Int,
            kMeansTolerance: Double,
            kMeansDistanceMeasurement: String,
            kMeansSeed: Long,
            kMeansPredictionCol: String,
            lshHashTables: Int,
            lshSeed: Long,
            lshOutputCol: String,
            quorumCount: Int,
            minimumVectorCountToMutate: Int,
            vectorMutationMethod: String,
            mutationMode: String,
            mutationValue: Double,
            labelBalanceMode: String,
            cardinalityThreshold: Int,
            numericRatio: Double,
            numericTarget: Int): DataFrame =
    new SyntheticFeatureGenerator(data)
      .setFeaturesCol(featuresCol)
      .setLabelCol(labelCol)
      .setSyntheticCol(syntheticCol)
      .setFieldsToIgnore(fieldsToIgnore)
      .setKGroups(kGroups)
      .setKMeansMaxIter(kMeansMaxIter)
      .setKMeansTolerance(kMeansTolerance)
      .setKMeansDistanceMeasurement(kMeansDistanceMeasurement)
      .setKMeansSeed(kMeansSeed)
      .setKMeansPredictionCol(kMeansPredictionCol)
      .setLSHHashTables(lshHashTables)
      .setLSHSeed(lshSeed)
      .setLSHOutputCol(lshOutputCol)
      .setQuorumCount(quorumCount)
      .setMinimumVectorCountToMutate(minimumVectorCountToMutate)
      .setVectorMutationMethod(vectorMutationMethod)
      .setMutationMode(mutationMode)
      .setMutationValue(mutationValue)
      .setLabelBalanceMode(labelBalanceMode)
      .setCardinalityThreshold(cardinalityThreshold)
      .setNumericRatio(numericRatio)
      .setNumericTarget(numericTarget)
      .upSample()
}
