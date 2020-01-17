package com.databricks.labs.automl.sanitize

import com.databricks.labs.automl.params.PearsonPayload
import com.databricks.labs.automl.utils.DataValidation
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

/**
  *
  * @param df                   : DataFrame -> Dataset with a vectorized field of features,
  *                             the feature columns, and a label column.
  * @param featureColumnListing : Array[String] -> List of all fields that make up the feature vector
  *
  *                             Usage:
  *                             val autoFiltered = new PearsonFiltering(featurizedData, fields)
  *                             .setLabelCol("label")
  *                             .setFeaturesCol("features")
  *                             .setFilterStatistic("pearsonStat")
  *                             .setFilterDirection("greater")
  *                             .setFilterMode("auto")
  *                             .setAutoFilterNTile(0.5)
  *                             .filterFields
  */
class PearsonFiltering(df: DataFrame,
                       featureColumnListing: Array[String],
                       modelType: String)
    extends DataValidation
    with SanitizerDefaults {

  private final val PRODUCT = "product"
  private final val COV_VALUE = "cov_calculation"
  private final val DEVIATION = "_deviation"
  private final val SQUARED = "_squared"

  private var _labelCol: String = defaultLabelCol
  private var _featuresCol: String = defaultFeaturesCol
  private var _filterStatistic: String = defaultPearsonFilterStatistic
  private var _filterDirection: String = defaultPearsonFilterDirection

  private var _filterManualValue: Double = defaultPearsonFilterManualValue
  private var _filterMode: String = defaultPearsonFilterMode
  private var _autoFilterNTile: Double = defaultPearsonAutoFilterNTile
  private var _parallelism: Int = 20

  final private val _dataFieldNames = df.schema.fieldNames
  final private val _dataFieldTypes = df.schema.fields

  def setLabelCol(value: String): this.type = {
    require(
      _dataFieldNames.contains(value),
      s"Label Field $value is not in DataFrame Schema."
    )
    _labelCol = value
    this
  }

  def setFeaturesCol(value: String): this.type = {
    require(
      _dataFieldNames.contains(value),
      s"Feature Field $value is not in DataFrame Schema."
    )
    require(
      _dataFieldTypes.filter(_.name == value)(0).dataType.typeName == "vector",
      s"Feature Field $value is not of vector type."
    )
    _featuresCol = value
    this
  }

  def setFilterStatistic(value: String): this.type = {
    require(
      _allowedStats.contains(value),
      s"Pearson Filtering Statistic '$value' is not a valid member of ${invalidateSelection(value, _allowedStats)}"
    )
    _filterStatistic = value
    this
  }

  def setFilterDirection(value: String): this.type = {
    require(
      _allowedFilterDirections.contains(value),
      s"Filter Direction '$value' is not a valid member of ${invalidateSelection(value, _allowedFilterDirections)}"
    )
    _filterDirection = value
    this
  }

  def setFilterManualValue(value: Double): this.type = {
    _filterManualValue = value
    this
  }

  def setFilterManualValue(value: Int): this.type = {
    _filterManualValue = value.toDouble
    this
  }

  def setFilterMode(value: String): this.type = {
    require(
      _allowedFilterModes.contains(value),
      s"Filter Mode $value is not a valid member of ${invalidateSelection(value, _allowedFilterModes)}"
    )
    _filterMode = value
    this
  }

  def setAutoFilterNTile(value: Double): this.type = {
    require(value <= 1.0 & value >= 0.0, "NTile value must be between 0 and 1.")
    _autoFilterNTile = value
    this
  }

  def setParallelism(value: Int): this.type = {
    _parallelism = value
    this
  }

  def getLabelCol: String = _labelCol
  def getFeaturesCol: String = _featuresCol
  def getFilterStatistic: String = _filterStatistic
  def getFilterDirection: String = _filterDirection
  def getFilterManualValue: Double = _filterManualValue
  def getFilterMode: String = _filterMode
  def getAutoFilterNTile: Double = _autoFilterNTile
  def getParallelism: Int = _parallelism

  private var _pearsonVectorFields: Array[String] = Array.empty
  private var _pearsonNonCategoricalFields: Array[String] = Array.empty

  private def setPearsonNonCategoricalFields(
    value: Array[String]
  ): this.type = {
    _pearsonNonCategoricalFields = value
    this
  }

  private def setPearsonVectorFields(value: Array[String]): this.type = {
    _pearsonVectorFields = value
    this
  }

  /**
    * Private method for calculating the ChiSq relation of each feature to the label column.
    * @param data DataFrame that contains the vector to test and the label column.
    * @param featureColumn the name of the feature column vector to be used in the test.
    * @return List of the stats from the comparison calculated.
    */
  private def buildChiSq(data: DataFrame,
                         featureColumn: String): List[PearsonPayload] = {
    val reportBuffer = new ListBuffer[PearsonPayload]

    val chi = ChiSquareTest.test(data, featureColumn, _labelCol).head
    val pvalues = chi.getAs[Vector](0).toArray
    val degreesFreedom = chi.getSeq[Int](1).toArray
    val pearsonStat = chi.getAs[Vector](2).toArray

    for (i <- _pearsonVectorFields.indices) {
      reportBuffer += PearsonPayload(
        _pearsonVectorFields(i),
        pvalues(i),
        degreesFreedom(i),
        pearsonStat(i)
      )
    }
    reportBuffer.result
  }

  /**
    * Method for, given a particular column, get the exact count of the cardinality of the field.
    * @param column Name of the column that is being tested for cardinality
    * @return [Long] the number of unique entries in the column
    */
  private def acquireCardinality(column: String): Long = {

    val aggregateData =
      df.select(col(column)).groupBy(col(column)).agg(count(col(column)))
    aggregateData.count()
  }

  /**
    * Private method for running through all of the fields included in the base feature vector and calculating their
    * cardinality in parallel (10x concurrency)
    * @return An Array of Field Name, Distinct Count
    */
  private def featuresCardinality(): Array[(String, Long)] = {

    val cardinalityOfFields = new ArrayBuffer[(String, Long)]()

    val featurePool = featureColumnListing.par
    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(_parallelism))
    featurePool.tasksupport = taskSupport

    featurePool.foreach { x =>
      cardinalityOfFields += Tuple2(x, acquireCardinality(x))
    }

    cardinalityOfFields.result.toArray
  }

  /**
    * Private method for analyzing the input feature vector columns, determining their cardinality, and updating
    * the private var's to use these new lists.
    * @return Nothing - it updates the class-scoped variables when called.
    */
  private def restrictFeatureSet(): this.type = {

    // Empty ArrayBuffer to hold the fields to build the PearsonFeature Vector
    val pearsonVectorBuffer = new ArrayBuffer[String]
    val pearsonNonCategoricalBuffer = new ArrayBuffer[String]

    val determineCardinality = featuresCardinality()

    determineCardinality.foreach { x =>
      if (x._2 < 50) pearsonVectorBuffer += x._1
      else pearsonNonCategoricalBuffer += x._1
    }

    setPearsonNonCategoricalFields(pearsonNonCategoricalBuffer.result.toArray)
    setPearsonVectorFields(pearsonVectorBuffer.result.toArray)

  }

  /**
    * Method for creating a new temporary feature vector that will be used for Pearson Filtering evaluation, removing
    * the high cardinality fields from this test.
    * @return [DataFrame] the DataFrame with a new vector entiitled "pearsonVector" that is used for removing
    *         fields from the feature vector that are either highly positively or negatively correlated to the label
    *         field.
    */
  private def reVectorize(): DataFrame = {

    // Create a new feature vector based on the fields that will be evaluated in PearsonFiltering
    restrictFeatureSet()

    require(
      _pearsonVectorFields.nonEmpty,
      s"Pearson Filtering contains all continuous variables in the feature" +
        s" vector, or cardinality of all features is greater than the threshold of 10k unique entries.  " +
        s"Please turn off pearson filtering for this data set by defining the main class with the setter: " +
        s".pearsonFilterOff() to continue."
    )

    val assembler = new VectorAssembler()
      .setInputCols(_pearsonVectorFields)
      .setOutputCol("pearsonVector")

    assembler.transform(df)
  }

  /**
    * Method for manually filtering out fields from the feature vector based on a user-supplied or
    * automation-calculated threshold cutoff.
    * @param statPayload the calculated correlation stats from feature elements in the vector to the label column.
    * @param filterValue the cut-off value specified by the user, or calculated through the quantile generator
    *                    methodology.
    * @return A list of fields that will be persisted and included in the feature vector going forward.
    */
  private def filterChiSq(statPayload: List[PearsonPayload],
                          filterValue: Double): List[String] = {
    val fieldRestriction = new ListBuffer[String]
    _filterDirection match {
      case "greater" =>
        statPayload.foreach(x => {
          x.getClass.getDeclaredFields foreach { f =>
            f.setAccessible(true)
            if (f.getName == _filterStatistic)
              if (f.get(x).asInstanceOf[Double] >= filterValue)
                fieldRestriction += x.fieldName
              else None
            else None
          }
        })
      case "lesser" =>
        statPayload.foreach(x => {
          x.getClass.getDeclaredFields foreach { f =>
            f.setAccessible(true)
            if (f.getName == _filterStatistic)
              if (f.get(x).asInstanceOf[Double] <= filterValue)
                fieldRestriction += x.fieldName
              else None
            else None
          }
        })
      case _ =>
        throw new UnsupportedOperationException(
          s"${_filterDirection} is not supported for manualFilterChiSq"
        )
    }
    fieldRestriction.result
  }

  /**
    * Method for automatically detecting the quantile values for the filter statistic to cull fields automatically
    * based on the distribution of correlation amongst the feature vector and the label.
    * @param pearsonResults The pearson (and other) stats that have been calculated between each element of the
    *                       feature vector and the label.
    * @return The PearsonPayload results for each field, filtering out those elements that are either above / below
    *         the threshold configured.
    */
  private def quantileGenerator(
    pearsonResults: List[PearsonPayload]
  ): Double = {

    val statBuffer = new ListBuffer[Double]
    pearsonResults.foreach(x => {
      x.getClass.getDeclaredFields foreach { f =>
        f.setAccessible(true)
        if (f.getName == _filterStatistic)
          statBuffer += f.get(x).asInstanceOf[Double]
      }
    })

    val statSorted = statBuffer.result.sortWith(_ < _)
    if (statSorted.size % 2 == 1)
      statSorted((statSorted.size * _autoFilterNTile).toInt)
    else {
      val splitLoc = math.floor(statSorted.size * _autoFilterNTile).toInt
      val splitCheck = if (splitLoc < 1) 1 else splitLoc.toInt
      val (high, low) = statSorted.splitAt(splitCheck)
      (high.last + low.head) / 2
    }

  }

  private def filterClassifier(
    ignoreFields: Array[String] = Array.empty[String]
  ): DataFrame = {

    val revectoredData = reVectorize()

    val chiSqData = buildChiSq(revectoredData, "pearsonVector")
    val featureFields: List[String] = _filterMode match {
      case "manual" =>
        filterChiSq(chiSqData, _filterManualValue)
      case _ =>
        filterChiSq(chiSqData, quantileGenerator(chiSqData))
    }
    require(
      featureFields.nonEmpty,
      "All feature fields have been filtered out.  Adjust parameters."
    )
    val fieldListing = featureFields ::: List(_labelCol) ::: ignoreFields.toList ::: _pearsonNonCategoricalFields.toList
    df.select(fieldListing.map(col): _*)

  }

  /**
    * Main entry point for Pearson Filtering
    * @param ignoreFields Fields that will be ignored from running a Pearson filter against.
    * @return
    */
  def filterFields(
    ignoreFields: Array[String] = Array.empty[String]
  ): DataFrame = {

    // Perform check of regression vs classification
    val uniqueLabelCounts = df
      .select(_labelCol)
      .agg(count(_labelCol).alias("uniques"))
      .first()
      .getAs[Long]("uniques")

    modelType match {
      case "classifier" => filterClassifier(ignoreFields)
      case _            => filterRegressor(ignoreFields)
    }

  }

  /**
    * Method for manually filtering out values whose linear correlation coefficient is greater than the _filterManualValue setting.
    * @param correlationData The mapping of each feature field's correlation and linear correlation coefficient valuest to the label
    * @return Field names that have not been filtered out
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def regressorManualFilter(
    correlationData: Map[String, (Double, Double)]
  ): Array[String] = {

    val fieldBuffer = new ArrayBuffer[String]
    correlationData.keys.foreach { x =>
      if (correlationData(x)._2 < _filterManualValue) fieldBuffer += x
    }
    fieldBuffer.toArray
  }

  /**
    * Method for doing quantile filtering (using the autoFilterNTile in automatic mode to filter out features that
    * show a linear correaltion coefficient that is greater than the autoFilterNTile value. (1.0 == perfect correlation)
    * @param correlationData The mapping of each feature field's correlation and linear correlation coefficient values to the label
    * @return Field names that have not been filtered out
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def regressionAutoFilter(
    correlationData: Map[String, (Double, Double)]
  ): Array[String] = {

    val fieldBuffer = new ArrayBuffer[String]
    correlationData.keys.foreach { x =>
      if (correlationData(x)._2 < _autoFilterNTile) fieldBuffer += x
    }
    fieldBuffer.toArray
  }

  /**
    * Method for filtering out a regression data set (detect extremely high collinearity in features compared to the label values
    * @param ignoreFields Fields to ignore from the test
    * @return Dataframe that has the highly correlated feature fields removed
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def filterRegressor(
    ignoreFields: Array[String] = Array.empty[String]
  ): DataFrame = {

    val featureFields = _filterMode match {
      case "manual" =>
        regressorManualFilter(calculateRegressionCovariance(ignoreFields))
      case _ =>
        regressionAutoFilter(calculateRegressionCovariance(ignoreFields))
    }

    require(
      featureFields.nonEmpty,
      "All feature fields have been filtered out.  Adjust parameters."
    )
    val fieldListing = featureFields.toList ::: List(_labelCol) ::: ignoreFields.toList
    df.select(fieldListing.map(col): _*)

  }

  /**
    * Private method for calculating the covariance and linear correlation coefficient for each feature field to the label
    * @param ignoreFields Fields to ignore in the analysis
    * @return Map of [FieldName, (Covariance value, Linear Correlation Coefficient)
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def calculateRegressionCovariance(
    ignoreFields: Array[String] = Array.empty[String]
  ) = {

    val summaryData = df
      .select(featureColumnListing ++ Array(_labelCol) map col: _*)
      .summary("mean")

    val rowCount = df
      .select(col(_labelCol))
      .agg(count(_labelCol).alias(_labelCol))
      .withColumn(_labelCol, col(_labelCol).cast(DoubleType))
      .first()
      .getAs[Double](_labelCol)

    val meanValues =
      summaryData.filter(col("summary") === "mean").drop("summary")
    val meanData =
      meanValues.first().getValuesMap[Double](meanValues.schema.fieldNames)

    val buffer = new ArrayBuffer[Map[String, (Double, Double)]]

    meanData.keys.foreach { x =>
      if (x != _labelCol)
        buffer += covarianceCalculation(x, meanData, rowCount)
    }

    buffer.result.flatten.toMap

  }

  /**
    * Private method for calculating the coveriance and linear correlation coefficient between a field and the label
    * @param field Field to compare
    * @param avgMap Mapping of the average values of each field and the label (calculated only once)
    * @param rowCount Double of the row count of the Dataframe (calculated only once)
    * @return Map of FieldName -> (covariance, linear correlation coefficient)
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def covarianceCalculation(
    field: String,
    avgMap: Map[String, Double],
    rowCount: Double
  ): Map[String, (Double, Double)] = {

    val tempDF = df
      .withColumn(field, col(field).cast(DoubleType))
      .select(field, _labelCol)
      .withColumn(field + DEVIATION, col(field) - avgMap(field))
      .withColumn(field + SQUARED, col(field) * col(field))
      .withColumn(_labelCol + DEVIATION, col(_labelCol) - avgMap(_labelCol))
      .withColumn(_labelCol + SQUARED, col(_labelCol) * col(_labelCol))
      .withColumn(
        COV_VALUE,
        col(field + DEVIATION) * col(_labelCol + DEVIATION)
      )
      .withColumn(PRODUCT, col(field) * col(_labelCol))

    val summed = tempDF
      .agg(
        sum(field).alias(field),
        sum(_labelCol).alias(_labelCol),
        sum(PRODUCT).alias(PRODUCT),
        sum(COV_VALUE).alias(COV_VALUE),
        sum(field + SQUARED).alias(field + SQUARED),
        sum(_labelCol + SQUARED).alias(_labelCol + SQUARED)
      )
      .first()
      .getValuesMap[Double](
        Seq(
          COV_VALUE,
          field,
          _labelCol,
          PRODUCT,
          _labelCol + SQUARED,
          field + SQUARED
        )
      )

    val linearCorrelationCoefficient = (summed(PRODUCT) - (summed(field) * summed(
      _labelCol
    ) / rowCount)) / math.sqrt(
      (summed(field + SQUARED) - math
        .pow(summed(field), 2.0) / rowCount) * (summed(_labelCol + SQUARED) - math
        .pow(summed(_labelCol), 2.0) / rowCount)
    )

    Map(field -> (summed(COV_VALUE) / rowCount, linearCorrelationCoefficient))

  }

}
