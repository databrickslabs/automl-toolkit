package com.databricks.labs.automl.sanitize

import com.databricks.labs.automl.exceptions.FeatureCorrelationException
import com.databricks.labs.automl.utils.SparkSessionWrapper
import com.databricks.labs.automl.utils.structures.{
  FieldCorrelationAggregationStats,
  FieldCorrelationPayload,
  FieldPairs,
  FieldRemovalPayload
}
import org.apache.log4j.Logger
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

class FeatureCorrelationDetection(data: DataFrame, fieldListing: Array[String])
    extends SparkSessionWrapper {

  private final val DEVIATION = "_deviation"
  private final val SQUARED = "_squared"
  private final val COV = "covariance_value"
  private final val PRODUCT = "_product"

  private val logger: Logger = Logger.getLogger(this.getClass)

  private var _correlationCutoffHigh: Double = 0.0
  private var _correlationCutoffLow: Double = 0.0
  private var _labelCol: String = "label"
  private var _parallelism: Int = 20

  final private val _dataFieldNames = data.schema.fieldNames

  def setCorrelationCutoffHigh(value: Double): this.type = {
    require(
      value <= 1.0,
      "Maximum range of Correlation Cutoff on the high end must be less than 1.0"
    )
    _correlationCutoffHigh = value
    this
  }

  def setCorrelationCutoffLow(value: Double): this.type = {
    require(
      value >= -1.0,
      "Minimum range of Correlation Cutoff on the low end must be greater than -1.0"
    )
    _correlationCutoffLow = value
    this
  }

  def setLabelCol(value: String): this.type = {
    require(
      _dataFieldNames.contains(value),
      s"Label field $value is not in Dataframe"
    )
    _labelCol = value
    this
  }

  def setParallelism(value: Int): this.type = {
    _parallelism = value
    this
  }

  def getCorrelationCutoffHigh: Double = _correlationCutoffHigh

  def getCorrelationCutoffLow: Double = _correlationCutoffLow

  def getLabelCol: String = _labelCol
  def getParallelism: Int = _parallelism

  def filterFeatureCorrelation(): DataFrame = {

    assert(
      _dataFieldNames.contains(_labelCol),
      s"Label field ${_labelCol} is not in Dataframe"
    )
    val featureCorrelation = determineFieldsToDrop
    data.drop(featureCorrelation.dropFields: _*)
  }

  /**
    * Create the left/right testing pairs to be used in determining correlation between feature fields
    * @return Array of distinct pairs of feature fields to test
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def buildFeaturePairs(): Array[FieldPairs] = {

    fieldListing
      .combinations(2)
      .map { case Array(x, y) => FieldPairs(x, y) }
      .toArray

  }

  /**
    * Method for calculating all of the pairwise correlation calculations for the feature fields
    * @return Array of FieldCorrelationPayload data (left/right name pairs and the correlation value)
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def calculateFeatureCorrelation: Array[FieldCorrelationPayload] = {

    val aggregationStats = getAggregationStats

    val interactionData = new ArrayBuffer[FieldCorrelationPayload]

    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(_parallelism))
    val pairs = buildFeaturePairs().par
    pairs.tasksupport = taskSupport

    pairs.foreach { x =>
      interactionData += calculateCorrelation(
        x,
        aggregationStats.rowCounts,
        aggregationStats.averageMap
      )
    }

    val preSort = interactionData.toArray.groupBy(_.primaryColumn)
    fieldListing.flatMap(x => preSort.get(x)).flatten

  }

  /**
    * Private method for determining the interaction combinations for the recursive pairwise comparison and
    * evaluating whether a particular column has positive or negative correlation to all other columns
    * and therefore should be dropped from the dataset.
    * @param data Array[FieldCorrelationPayload] that represents the pairwise correlation between fields
    * @return Map[String, Double] with each field and the percentage of other fields it meets the criteria for
    *         filtering based on the correlation cutoff values.
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def calculateGroupStats(
    data: Array[FieldCorrelationPayload]
  ): Map[String, Double] = {
    data.groupBy(_.primaryColumn).map {
      case (k, v) =>
        val positiveCounts =
          v.count(_.correlation >= _correlationCutoffHigh).toDouble
        val negativeCounts =
          v.count(_.correlation <= _correlationCutoffLow).toDouble
        k -> (positiveCounts + negativeCounts) / v.length
    }
  }

  /**
    * Method for determining which columns need to be dropped from the feature set based on the correlation cutoff settings
    * @return FieldRemovalPayload that contains the removal and retain fields.
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    * @throws FeatureCorrelationException: totalFields, removedFields
    */
  @throws(classOf[RuntimeException])
  def determineFieldsToDrop: FieldRemovalPayload = {

    val retainBuffer = mutable.SortedSet[String]()
    val removeBuffer = mutable.SortedSet[String]()

    val correlationResult = calculateFeatureCorrelation
    val groupData = calculateGroupStats(correlationResult)

    correlationResult.foreach(x => {

      if (!removeBuffer
            .contains(x.primaryColumn) && groupData(x.primaryColumn) < 1.0) {
        if (x.correlation >= _correlationCutoffHigh) {
          retainBuffer += x.primaryColumn
          removeBuffer += x.pairs.right
        } else if (x.correlation <= _correlationCutoffLow) {
          retainBuffer += x.primaryColumn
          removeBuffer += x.pairs.right
        } else {
          retainBuffer += x.primaryColumn
        }
      } else {
        removeBuffer += x.primaryColumn
      }

    })

    // Validation
    if (retainBuffer.isEmpty)
      throw FeatureCorrelationException(
        fieldListing,
        removeBuffer.result.toArray
      )
    FieldRemovalPayload(removeBuffer.toArray, retainBuffer.toArray)

  }

  /**
    * Debug method to allow for an inspection of the correlation between each feature value to one another
    * @return DataFrame that contains the pair information and the correlation values of those pairs
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateFeatureCorrelationReport: DataFrame = {

    import spark.sqlContext.implicits._

    sc.parallelize(calculateFeatureCorrelation).toDF

  }

  /**
    * Private method for accessing the required data needed for correlation calculation (one-time calculation)
    * @return FieldCorrelationAggregationStats of row counts and map of average values for each feature field
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def getAggregationStats: FieldCorrelationAggregationStats = {

    val summaryMap = data
      .select(fieldListing map col: _*)
      .summary("mean")
      .filter(col("summary") === "mean")
      .drop("summary")
      .first()
      .getValuesMap[Double](fieldListing)

    val rowCounts = data
      .select(_labelCol)
      .agg(count(_labelCol).alias(_labelCol))
      .withColumn(_labelCol, col(_labelCol).cast(DoubleType))
      .first()
      .getAs[Double](_labelCol)
    FieldCorrelationAggregationStats(rowCounts, summaryMap)
  }

  /**
    * Private method for executing a pair-wise correlation calculation between two feature fields
    * @param pair the left/right pair of feature fields to calculate
    * @param rowCount Number of rows within the data set
    * @param averages average value of each feature as a Map
    * @return FieldCorrelationPayload that contains the pair and the correlation value
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def calculateCorrelation(
    pair: FieldPairs,
    rowCount: Double,
    averages: Map[String, Double]
  ): FieldCorrelationPayload = {

    // Establish a subset DataFrame that contains only the two columns being tested
    val subsetDataFrame = data
      .select(pair.left, pair.right)
      .withColumn(pair.left, col(pair.left).cast(DoubleType))
      .withColumn(pair.right, col(pair.right).cast(DoubleType))

    // Get the map of the values required to support the linear correlation calculation
    val covarianceMap = subsetDataFrame
      .withColumn(pair.left + DEVIATION, col(pair.left) - averages(pair.left))
      .withColumn(
        pair.right + DEVIATION,
        col(pair.right) - averages(pair.right)
      )
      .withColumn(pair.left + SQUARED, pow(col(pair.left), 2))
      .withColumn(pair.right + SQUARED, pow(col(pair.right), 2))
      .withColumn(COV, col(pair.left + DEVIATION) * col(pair.right + DEVIATION))
      .withColumn(PRODUCT, col(pair.left) * col(pair.right))
      .agg(
        sum(pair.left).alias(pair.left),
        sum(pair.right).alias(pair.right),
        sum(PRODUCT).alias(PRODUCT),
        sum(COV).alias(COV),
        sum(pair.left + SQUARED).alias(pair.left + SQUARED),
        sum(pair.right + SQUARED).alias(pair.right + SQUARED)
      )
      .first()
      .getValuesMap[Double](
        Seq(
          COV,
          pair.left,
          pair.right,
          PRODUCT,
          pair.left + SQUARED,
          pair.right + SQUARED
        )
      )

    // Calculate the correlation
    val linearCorrelationCoefficient = (covarianceMap(PRODUCT) - (covarianceMap(
      pair.left
    ) * covarianceMap(pair.right) / rowCount)) /
      math.sqrt(
        (covarianceMap(pair.left + SQUARED) - math
          .pow(covarianceMap(pair.left), 2.0) / rowCount) * (covarianceMap(
          pair.right + SQUARED
        ) - math.pow(covarianceMap(pair.right), 2.0) / rowCount)
      )

    FieldCorrelationPayload(pair.left, pair, linearCorrelationCoefficient)

  }

}
