package com.databricks.labs.automl.model.tools.split

import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, count, lit, row_number}
import org.apache.spark.sql.types.StructType
import org.apache.spark.storage.StorageLevel

object SplitOperators extends SparkSessionWrapper {

  @transient private val logger: Logger = Logger.getLogger(this.getClass)

  private def toDoubleType(x: Any): Option[Double] = x match {
    case i: Int    => Some(i)
    case d: Double => Some(d)
    case _         => None
  }

  private def generateEmptyTrainTest(
    schema: StructType
  ): (DataFrame, DataFrame) = {

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
                      uniqueLabels: Array[Row],
                      labelCol: String,
                      trainPortion: Double): Array[DataFrame] = {

    logger.log(Level.DEBUG, "DEBUG: Generating empty train/test split sets")

    var (trainData, testData) = generateEmptyTrainTest(data.schema)

    uniqueLabels.foreach { x =>
      logger.log(Level.DEBUG, s"DEBUG: Unique Label: $x")

      val conversionValue = toDoubleType(x(0)).get

      val Array(trainSplit, testSplit) = data
        .filter(col(labelCol) === conversionValue)
        .randomSplit(Array(trainPortion, 1 - trainPortion), seed)

      trainData = trainData.union(trainSplit)
      testData = testData.union(testSplit)

      logger.log(Level.DEBUG, "DEBUG: returning train & test datasets")
    }

    Array(trainData, testData)
  }

  def underSampleSplit(data: DataFrame,
                       seed: Long,
                       labelCol: String,
                       trainPortion: Double): Array[DataFrame] = {

    logger.log(Level.DEBUG, "DEBUG: Generating empty train/test split sets")

    var (trainData, testData) = generateEmptyTrainTest(data.schema) 

    val totalDataSetCount = data.count()

    val groupedLabelCount = data
      .select(labelCol)
      .groupBy(labelCol)
      .agg(count("*").as("counts"))
      .withColumn("skew", col("counts") / lit(totalDataSetCount))
      .select(labelCol, "skew")

    val uniqueGroups = groupedLabelCount.collect()

    val smallestSkew = groupedLabelCount
      .sort(col("skew").asc)
      .select(col("skew"))
      .first()
      .getDouble(0)

    uniqueGroups.foreach { x =>
      logger.log(Level.DEBUG, s"DEBUG: Unique Label: $x")

      val groupData = toDoubleType(x(0)).get

      val groupRatio = x.getDouble(1)

      val groupDataFrame = data.filter(col(labelCol) === groupData)

      val Array(train, test) = if (groupRatio == smallestSkew) {
        groupDataFrame.randomSplit(Array(trainPortion, 1 - trainPortion), seed)
      } else {
        groupDataFrame
          .sample(withReplacement = false, smallestSkew / groupRatio)
          .randomSplit(Array(trainPortion, 1 - trainPortion), seed)
      }

      trainData = trainData.union(train)
      testData = testData.union(test)

    }

    logger.log(Level.DEBUG, "DEBUG: returning train & test datasets")

    Array(trainData, testData)

  }

  def overSampleSplit(data: DataFrame,
                      seed: Long,
                      labelCol: String,
                      trainPortion: Double): Array[DataFrame] = {

    logger.log(Level.DEBUG, "DEBUG: Generating empty train/test split sets")

    var (trainData, testData) = generateEmptyTrainTest(data.schema)

    val groupedLabelCount = data
      .select(labelCol)
      .groupBy(labelCol)
      .agg(count("*").as("counts"))

    val uniqueGroups = groupedLabelCount.collect()

    val largestGroupCount = groupedLabelCount
      .sort(col("counts").desc)
      .select(col("counts"))
      .first()
      .getLong(0)

    uniqueGroups.foreach { x =>
      logger.log(Level.DEBUG, s"DEBUG: Unique Label: $x")

      val groupData = toDoubleType(x(0)).get

      val groupRatio = math.ceil(largestGroupCount / x.getLong(1)).toInt

      for (i <- 1 to groupRatio) {

        val Array(train, test): Array[DataFrame] = data
          .filter(col(labelCol) === groupData)
          .randomSplit(Array(trainPortion, 1 - trainPortion), seed)

        trainData = trainData.union(train)
        testData = testData.union(test)

      }
    }

    logger.log(Level.DEBUG, "DEBUG: returning train & test datasets")

    Array(trainData, testData)

  }

  def stratifyReduce(data: DataFrame,
                     reductionFactor: Double,
                     seed: Long,
                     uniqueLabels: Array[Row],
                     labelCol: String,
                     trainPortion: Double): Array[DataFrame] = {

    logger.log(Level.DEBUG, "DEBUG: Generating empty train/test split sets")

    var (trainData, testData) = generateEmptyTrainTest(data.schema)

    uniqueLabels.foreach { x =>
      logger.log(Level.DEBUG, s"DEBUG: Unique Label: $x")

      val conversionValue = toDoubleType(x(0)).get

      val Array(trainSplit, testSplit) = data
        .filter(col(labelCol) === conversionValue)
        .randomSplit(Array(trainPortion, 1 - trainPortion), seed)

      trainData = trainData.union(trainSplit.sample(reductionFactor))
      testData = testData.union(testSplit.sample(reductionFactor))

    }

    logger.log(Level.DEBUG, "DEBUG: returning train & test datasets")

    Array(trainData, testData)

  }

  def chronologicalSplit(data: DataFrame,
                         seed: Long,
                         trainSplitChronologicalColumn: String,
                         trainSplitChronologicalRandomPercentage: Double,
                         trainPortion: Double): Array[DataFrame] = {

    require(
      data.schema.fieldNames.contains(trainSplitChronologicalColumn),
      s"Chronological Split Field ${trainSplitChronologicalColumn} is not in schema: " +
        s"${data.schema.fieldNames.mkString(", ")}"
    )

    // Validation check for the random 'wiggle value' if it's set that it won't risk creating zero rows in train set.
    if (trainSplitChronologicalRandomPercentage > 0.0)
      require(
        (1 - trainPortion) * trainSplitChronologicalRandomPercentage / 100 < 0.5,
        s"With trainSplitChronologicalRandomPercentage set at '${trainSplitChronologicalRandomPercentage}' " +
          s"and a train test ratio of ${trainPortion} there is a high probability of train data set being empty." +
          s"  \n\tAdjust lower to prevent non-deterministic split levels that could break training."
      )

    // Get the row count
    val rawDataCount = data.count.toDouble

    val splitValue = scala.math.round(rawDataCount * trainPortion).toInt

    // Get the row number estimation for conduction the split at
    val splitRow: Int = if (trainSplitChronologicalRandomPercentage <= 0.0) {
      splitValue
    } else {
      // randomly mutate the size of the test validation set
      val splitWiggle = scala.math
        .round(
          rawDataCount * (1 - trainPortion) *
            trainSplitChronologicalRandomPercentage / 100
        )
        .toInt
      splitValue - scala.util.Random.nextInt(splitWiggle)
    }

    // Define the window partition
    val uniqueCol = "chron_grp_autoML_" + java.util.UUID.randomUUID().toString

    // Define temporary non-colliding columns for the window partition
    val uniqueRow = "row_" + java.util.UUID.randomUUID().toString
    val windowDefintion =
      Window.partitionBy(uniqueCol).orderBy(trainSplitChronologicalColumn)

    // Generate a new Dataframe that has the row number partition, sorted by the chronological field
    val preSplitData = data
      .withColumn(uniqueCol, lit("grp"))
      .withColumn(uniqueRow, row_number() over windowDefintion)
      .drop(uniqueCol)

    logger.log(Level.DEBUG, "DEBUG: returning train & test datasets")

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
                     uniqueLabels: Array[Row],
                     syntheticCol: String,
                     labelCol: String,
                     trainPortion: Double): Array[DataFrame] = {

    logger.log(Level.DEBUG, "DEBUG: generating KSample data sets")

    // Split out the real from the synthetic data
    val realData = data.filter(!col(syntheticCol))

    // Split out the synthetic data
    val syntheticData = data.filter(col(syntheticCol))

    // Perform stratified splits on both the real and synthetic data
    val Array(realTrain, realTest) =
      stratifiedSplit(realData, seed, uniqueLabels, labelCol, trainPortion)

    val Array(syntheticTrain, syntheticTest) =
      stratifiedSplit(syntheticData, seed, uniqueLabels, labelCol, trainPortion)

    logger.log(
      Level.DEBUG,
      "DEBUG: returning data sets augmented with KSample synthetic data"
    )

    // Union the real train data with the synthetic train data and return that with only the real test data
    Array(realTrain.union(syntheticTrain), realTest)

  }

  def genTestTrain(data: DataFrame,
                   seed: Long,
                   uniqueLabels: Array[Row],
                   trainSplitMethod: String,
                   labelCol: String,
                   trainPortion: Double,
                   syntheticCol: String = "syntheticColumn",
                   trainSplitChronologicalColumn: String = "datetime",
                   trainSplitChronologicalRandomPercentage: Double = 0.05,
                   reductionFactor: Double = 0.5): Array[DataFrame] = {

    logger.log(Level.DEBUG, s"DEBUG: Split Method: ${trainSplitMethod}")

    trainSplitMethod match {
      case "random" =>
        data.randomSplit(Array(trainPortion, 1 - trainPortion), seed)
      case "chronological" =>
        chronologicalSplit(
          data,
          seed,
          trainSplitChronologicalColumn,
          trainSplitChronologicalRandomPercentage,
          trainPortion
        )
      case "stratified" =>
        stratifiedSplit(data, seed, uniqueLabels, labelCol, trainPortion)
      case "overSample"  => overSampleSplit(data, seed, labelCol, trainPortion)
      case "underSample" => underSampleSplit(data, seed, labelCol, trainPortion)
      case "stratifyReduce" =>
        stratifyReduce(
          data,
          reductionFactor,
          seed,
          uniqueLabels,
          labelCol,
          trainPortion
        )
      case "kSample" =>
        kSamplingSplit(
          data,
          seed,
          uniqueLabels,
          syntheticCol,
          labelCol,
          trainPortion
        )
      case _ =>
        throw new IllegalArgumentException(
          s"Cannot conduct train test split in mode: '${trainSplitMethod}'"
        )
    }

  }

  def optimizeTestTrain(train: DataFrame,
                        test: DataFrame,
                        optimalParts: Int,
                        shuffle: Boolean = false): (DataFrame, DataFrame) = {
    //  TODO: TOMES - Why is this still hardocded DISK_ONLY?
    logger.log(
      Level.DEBUG,
      s"DEBUG: Train persist called. Shuffle = $shuffle. Optimal parts: $optimalParts"
    )
    val optimizedTrain = if (shuffle) {
      train.repartition(optimalParts).persist(StorageLevel.DISK_ONLY)
    } else {
      train.coalesce(optimalParts).persist(StorageLevel.DISK_ONLY)
    }

    logger.log(
      Level.DEBUG,
      s"DEBUG: Test persist called. Shuffle = $shuffle. Optimal parts: $optimalParts"
    )
    val optimizedTest = if (shuffle) {
      test.repartition(optimalParts).persist(StorageLevel.DISK_ONLY)
    } else {
      test.coalesce(optimalParts).persist(StorageLevel.DISK_ONLY)
    }

    logger.log(Level.DEBUG, "DEBUG: Forcing the persist for Train")
    optimizedTrain.foreach(_ => ())
    logger.log(Level.DEBUG, "DEBUG: Forcing the persist for Test")
    optimizedTest.foreach(_ => ())

    (optimizedTrain, optimizedTest)

  }

}
