package com.databricks.labs.automl.utilities

import org.joda.time.LocalDate
import com.databricks.labs.automl.utils.structures.ArrayGeneratorMode
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, when}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

trait DataGeneratorUtilities {

  final val DOUBLE_FILL = Double.MinValue
  final val FLOAT_FILL = Float.MinValue
  final val INT_FILL = Int.MinValue
  final val LONG_FILL = Long.MinValue

  import com.databricks.labs.automl.utils.structures.ArrayGeneratorMode._

  final case class ArrayGeneratorException(
    private val mode: String,
    private val allowableModes: Array[String],
    cause: Throwable = None.orNull
  ) extends RuntimeException(
        s"The array generator mode " +
          s"specified: $mode is not in the allowable list of supported models: ${allowableModes
            .mkString(", ")}",
        cause
      )

  case class ModuloResult(factor: Int, remain: Int)

  /**
    * Enumeration assignment for array sorting mode
    * @param mode String - one of 'ascending', 'descending' or 'random'
    * @return Enumerated value
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def getArrayMode(mode: String): ArrayGeneratorMode.Value = {
    mode match {
      case "ascending"  => ASC
      case "descending" => DESC
      case "random"     => RAND
      case _ =>
        throw ArrayGeneratorException(
          mode,
          Array("random", "descending", "ascending")
        )
    }
  }

  /**
    * Helper method for getting the module quotient and remainder
    * @param x Integer value
    * @param y Integer divisor value
    * @return ModuleResult of the quotient factor and the remainder from the division
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def getRemainder(x: Int, y: Int): ModuloResult = {
    import scala.math.Integral.Implicits._
    val (q, r) = x /% y
    ModuloResult(q, r)
  }

  /**
    * Method for generating String Arrays of arbitrary size and uniqueness
    * @param targetCount number of string elements to generate
    * @param uniqueValueCount the number of unique elements within the collection
    * @return Array of Strings
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateStringData(targetCount: Int,
                         uniqueValueCount: Int): Array[String] = {

    val orderedColl = ('a' to 'z').toArray.map(_.toString)

    val repeatingChain = getRemainder(targetCount, orderedColl.length)

    val uniqueBuffer = ArrayBuffer[String](orderedColl: _*)

    if (uniqueValueCount > orderedColl.length) {
      for (x <- 0 until repeatingChain.factor) {
        orderedColl.foreach(y => uniqueBuffer += y + x.toString)
      }
    }

    val uniqueArray = uniqueBuffer.take(uniqueValueCount).toArray

    val outputArray = if (uniqueValueCount > orderedColl.length) {
      Array
        .fill(repeatingChain.factor)(uniqueArray)
        .flatten ++ uniqueArray
        .take(repeatingChain.remain)
    } else {
      Array.fill(targetCount / (uniqueValueCount - 1))(uniqueArray).flatten
    }

    outputArray.take(targetCount)
  }

  /**
    * Method for generating String Arrays with nullable values based on the supplied rate and offset
    * @param targetCount Count of Strings to generate in the Array
    * @param uniqueValueCount Number of unique / distinct Strings to have in the collection
    * @param targetNullRate The rate with which to generate null values in the Array (deterministic)
    * @param nullOffset The offset in which to start iterating through and nulling the values in the Array
    * @return Array[String] with nulls inserted.
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateStringDataWithNulls(targetCount: Int,
                                  uniqueValueCount: Int,
                                  targetNullRate: Int,
                                  nullOffset: Int): Array[String] = {

    generateStringData(targetCount, uniqueValueCount).zipWithIndex.map {
      case (v, i) => if ((i + nullOffset) % targetNullRate != 0.0) v else null
    }

  }

  /**
    * to simulate a non-linear distribution for validation tests in handling the data in Feature Engineering tests
    * @param targetCount Number of values to generate
    * @return Array of Doubles in the Fibonacci sequence
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateFibonacciData(targetCount: Int): Array[Double] = {
    lazy val fibs
      : Stream[BigInt] = BigInt(0) #:: BigInt(1) #:: (fibs zip fibs.tail)
      .map(x => x._1 + x._2)
    fibs.take(targetCount).toArray.map(_.toDouble)
  }

  /**
    * Method for generating a Fibonacci sequence with a fill condition in order to nullify later
    * @param targetCount number of elements in the Fibonacci sequence to generate
    * @param targetNullRate Frequency of null values to generate
    * @param nullOffset first index position to start filling in the fillable nullable values
    * @return Array of Double of the Fibonacci sequence
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateFibonacciDataWithNulls(targetCount: Int,
                                     targetNullRate: Int,
                                     nullOffset: Int): Array[Double] = {
    generateFibonacciData(targetCount).zipWithIndex.map {
      case (v, i) =>
        if ((i + nullOffset) % targetNullRate != 0.0) v else DOUBLE_FILL
    }
  }

  /**
    * Method for generating linear-distributed Doubles
    * @param targetCount Desired number of doubles in the array
    * @param start starting point for data generation in the Range function
    * @param step the value of distance between the start and the target count value
    * @param mode for the DataFrame's by row reference of sorting order of the Array
    * @return Array of Doubles
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateDoublesData(targetCount: Int,
                          start: Double,
                          step: Double,
                          mode: String): Array[Double] = {

    val sortMode = getArrayMode(mode)

    val stoppingPoint = (targetCount * step) + start

    val doubleArray =
      Range.BigDecimal(start, stoppingPoint, step).toArray.map(_.toDouble)

    sortMode match {
      case ASC  => doubleArray.sortWith(_ < _)
      case DESC => doubleArray.sortWith(_ > _)
      case RAND => Random.shuffle(doubleArray.toList).toArray
    }
  }

  /**
    * Method for generating a series of Doubles with a fill condition in order to nullify later
    * @param targetCount Desired number of doubles to generate
    * @param start Starting position
    * @param step Space between each value
    * @param mode sorting mode
    * @param targetNullRate Frequency of null values to generate
    * @param nullOffset first index position to start from to generate nulls
    * @return Array of Doubles with fillable values in place where nulls will be in the test DataFrame
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateDoublesDataWithNulls(targetCount: Int,
                                   start: Double,
                                   step: Double,
                                   mode: String,
                                   targetNullRate: Int,
                                   nullOffset: Int): Array[Double] = {

    generateDoublesData(targetCount, start, step, mode).zipWithIndex.map {
      case (v, i) =>
        if ((i + nullOffset) % targetNullRate != 0.0) v else DOUBLE_FILL
    }

  }

  /**
    * Method for generating ordinal Double Data (repeating values)
    * @param targetCount Total number of Doubles to generate in the series
    * @param start Starting point for the repeating series
    * @param step Distance between the repeating series values
    * @param mode sorting mode for the repeating arrays
    * @param distinctValues number of elements in the repeating series
    * @return Array[Double] of Repeating Ordinal Doubles
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateRepeatingDoublesData(targetCount: Int,
                                   start: Double,
                                   step: Double,
                                   mode: String,
                                   distinctValues: Int): Array[Double] = {

    val sortMode = getArrayMode(mode)
    val subStopPoint = (distinctValues * step) + start - 1.0
    val distinctArray = (start to subStopPoint by step).toArray
    val sortedArray = sortMode match {
      case ASC  => distinctArray.sortWith(_ < _)
      case DESC => distinctArray.sortWith(_ > _)
      case RAND => distinctArray
    }
    val outputArray = Array
      .fill(targetCount / (sortedArray.length - 1))(sortedArray)
      .flatten
      .take(targetCount)

    if (sortMode == RAND) Random.shuffle(outputArray.toList).toArray
    else outputArray

  }

  /**
    * Method for generating synthetic float series data
    * @param targetCount Number of floats to generate
    * @param start Starting offset position
    * @param step Offset between each element
    * @param mode sorting mode (ascending / descending /  random shuffle)
    * @return Array of Floats
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateFloatsData(targetCount: Int,
                         start: Float,
                         step: Float,
                         mode: String): Array[Float] = {

    val sortMode = getArrayMode(mode)
    val stoppingPoint = (targetCount * step) + start

    val floatArray = (start to stoppingPoint by step).toArray

    sortMode match {
      case ASC  => floatArray.sortWith(_ < _)
      case DESC => floatArray.sortWith(_ > _)
      case RAND => Random.shuffle(floatArray.toList).toArray
    }

  }

  /**
    * Method for generating an array of floats with a fill condition in order to nullify later
    * @param targetCount Number of floats to generate
    * @param start Starting offset position
    * @param step Offset between each element
    * @param mode sorting mode (ascending / descending /  random shuffle)
    * @param targetNullRate frequency of min val nullable values to generate
    * @param nullOffset first index position to start from to generate the null-fillable min values
    * @return Array of Float with fillable values in place where nulls will be converted in the test DataFrame
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateFloatsDataWithNulls(targetCount: Int,
                                  start: Float,
                                  step: Float,
                                  mode: String,
                                  targetNullRate: Int,
                                  nullOffset: Int): Array[Float] = {
    generateFloatsData(targetCount, start, step, mode).zipWithIndex.map {
      case (v, i) =>
        if ((i + nullOffset) % targetNullRate != 0.0) v else FLOAT_FILL
    }
  }

  /**
    * Method for generating a series of Doubles with a logarithmic distribution
    * @param targetCount Number of Doubles to generate
    * @param start starting min value for the linear series
    * @param step distance between values for the linear series that then gets converted into log scale
    * @param mode sorting mode
    * @return Array of Doubles in a logarithmic distribution
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateLogData(targetCount: Int,
                      start: Double,
                      step: Double,
                      mode: String): Array[Double] = {
    val doublesSequence = generateDoublesData(targetCount, start, step, mode)

    val minSeq =
      if (doublesSequence.min == 0.0) DOUBLE_FILL else doublesSequence.min
    val maxSeq = doublesSequence.max

    val b = math.log(maxSeq / minSeq) / (maxSeq - minSeq)
    val a = maxSeq / math.exp(b * maxSeq)

    doublesSequence.map { x =>
      a * math.exp(b * x)
    }

  }

  /**
    * Generate a log distribution with a fill condition in order to nullify later
    * @param targetCount Target number of elements to produce in a log distribution
    * @param start Starting min value of the distribution
    * @param step spacing on a linear series between values
    * @param mode sorting mode
    * @param targetNullRate frequency of fillable values for null replacement
    * @param nullOffset index of series to start the nullable MinValue elements for null replacement
    * @return Array of Doubles of a logarithmic distribution with Double.MinValue for null replacement
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateLogDataWithNulls(targetCount: Int,
                               start: Double,
                               step: Double,
                               mode: String,
                               targetNullRate: Int,
                               nullOffset: Int): Array[Double] = {
    generateLogData(targetCount, start, step, mode).zipWithIndex.map {
      case (v, i) =>
        if ((i + nullOffset) % targetNullRate != 0.0) v else DOUBLE_FILL
    }
  }

  /**
    * Generate a series of Doubles with an exponential distribution
    * @param targetCount Desired number of elements in the series
    * @param start starting point for the minimum value of the linear series
    * @param step numeric distance between each value of the linear series
    * @param mode sort mode for the series
    * @param power exponent that each element will be raised to
    * @return Array of Doubles that have been raised to a particular power
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateExponentialData(targetCount: Int,
                              start: Double,
                              step: Double,
                              mode: String,
                              power: Int): Array[Double] = {

    generateDoublesData(targetCount, start, step, mode).map(math.pow(_, power))

  }

  /**
    * Generate an exponential distributed Doubles series with a fill condition in order to nullify later
    * @param targetCount Desired number of elements in the series
    * @param start starting point for the minimum value of the linear series
    * @param step numeric distance between each value of the linear series
    * @param mode sort mode for the series
    * @param power exponent that each element will be raised to
    * @param targetNullRate frequency of fillable values for null replacement
    * @param nullOffset index of series to start the nullable MinValue elements for null replacement
    * @return Array of Doubles of a exponential distribution with Double.MinValue for null replacement
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateExponentialDataWithNulls(targetCount: Int,
                                       start: Double,
                                       step: Double,
                                       mode: String,
                                       power: Int,
                                       targetNullRate: Int,
                                       nullOffset: Int): Array[Double] = {

    generateExponentialData(targetCount, start, step, mode, power).zipWithIndex
      .map {
        case (v, i) =>
          if ((i + nullOffset) % targetNullRate != 0.0) v else DOUBLE_FILL
      }

  }

  /**
    * Method for generating a series of Integers, linearly distributed
    * @param targetCount Number of Integers to generate
    * @param start Starting Integer to work from
    * @param step numeric distance between each value in the linear series
    * @param mode sorting mode for the series (ascending, descending, or random)
    * @return Array of Integers with a linear distribution
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateIntData(targetCount: Int,
                      start: Int,
                      step: Int,
                      mode: String): Array[Int] = {

    val sortMode = getArrayMode(mode)
    val stoppingPoint = (targetCount * step) + start

    val intArray = (start to stoppingPoint by step).toArray

    sortMode match {
      case ASC  => intArray.sortWith(_ < _)
      case DESC => intArray.sortWith(_ > _)
      case RAND => Random.shuffle(intArray.toList).toArray
    }

  }

  /**
    * Method for generating a linear series of Integers with a fill condition in order to nullify later
    * @param targetCount Number of Integers to generate
    * @param start starting position for the series (min Int value)
    * @param step numeric distance between each value in the linear series
    * @param mode sorting mode for the series
    * @param targetNullRate frequency of fillable values to insert that will later be nulled
    * @param nullOffset adjustment to the starting position
    * @return Array of Integers in a linear distribution with nulls inserted
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateIntDataWithNulls(targetCount: Int,
                               start: Int,
                               step: Int,
                               mode: String,
                               targetNullRate: Int,
                               nullOffset: Int): Array[Int] = {

    generateIntData(targetCount, start, step, mode).zipWithIndex.map {
      case (v, i) =>
        if ((i + nullOffset) % targetNullRate != 0.0) v else INT_FILL
    }

  }

  /**
    * Method for generating series of Longs
    * @param targetCount Number of Long values to generate
    * @param start starting position for the series (minimum Long value)
    * @param step numeric distance between each value in the linear series
    * @param mode sorting mode for the series
    * @return Array[Long] in a linear distribution
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateLongData(targetCount: Int,
                       start: Long,
                       step: Long,
                       mode: String): Array[Long] = {
    val sortMode = getArrayMode(mode)
    val stoppingPoint = (targetCount * step) + start

    val longArray = (start to stoppingPoint by step).toArray

    sortMode match {
      case ASC  => longArray.sortWith(_ < _)
      case DESC => longArray.sortWith(_ > _)
      case RAND => Random.shuffle(longArray.toList).toArray
    }
  }

  /**
    * Method for generating a series of Long values with a fill condition in order to nullify later
    * @param targetCount Number of Long values to generate
    * @param start starting position for the series (minimum Long value)
    * @param step numeric distance between each value in the linear series
    * @param mode sorting mode for the series
    * @param targetNullRate rate of inserting fillable values to null later
    * @param nullOffset offset on starting position of series to place fillable values
    * @return Array[Long] with nullable values filled
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateLongDataWithNulls(targetCount: Int,
                                start: Long,
                                step: Long,
                                mode: String,
                                targetNullRate: Int,
                                nullOffset: Int): Array[Long] = {
    generateLongData(targetCount, start, step, mode).zipWithIndex.map {
      case (v, i) =>
        if ((i + nullOffset) % targetNullRate != 0.0) v else LONG_FILL
    }
  }

  /**
    * Method for generating ordinal Long Data (repeating values)
    * @param targetCount Total number of Longs to generate in the series
    * @param start Starting point for the repeating series
    * @param step Distance between the repeating series values
    * @param mode sorting mode for the repeating arrays
    * @param distinctValues number of elements in the repeating series
    * @return Array[Long] of Repeating Ordinal Longs
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateRepeatingLongData(targetCount: Int,
                                start: Long,
                                step: Long,
                                mode: String,
                                distinctValues: Int): Array[Long] = {

    val sortMode = getArrayMode(mode)
    val subStopPoint = (distinctValues * step) + start - 1L
    val distinctArray = (start to subStopPoint by step).toArray
    val sortedArray = sortMode match {
      case ASC  => distinctArray.sortWith(_ < _)
      case DESC => distinctArray.sortWith(_ > _)
      case RAND => distinctArray
    }
    val outputArray = Array
      .fill(targetCount / (sortedArray.length - 1))(sortedArray)
      .flatten
      .take(targetCount)

    if (sortMode == RAND) Random.shuffle(outputArray.toList).toArray
    else outputArray

  }

  /**
    * Method for generating ordinal Integer Data (repeating values)
    * @param targetCount Total number of Integers to generate in the series
    * @param start Starting point for the repeating series
    * @param step Distance between the repeating series values
    * @param mode sorting mode for the repeating arrays
    * @param distinctValues number of elements in the repeating series
    * @return Array[Int] of Repeating Ordinal Integers
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateRepeatingIntData(targetCount: Int,
                               start: Int,
                               step: Int,
                               mode: String,
                               distinctValues: Int): Array[Int] = {

    val sortMode = getArrayMode(mode)
    val subStopPoint = (distinctValues * step) + start - 1
    val distinctArray = (start to subStopPoint by step).toArray
    val sortedArray = sortMode match {
      case ASC  => distinctArray.sortWith(_ < _)
      case DESC => distinctArray.sortWith(_ > _)
      case RAND => distinctArray
    }
    val outputArray = Array
      .fill(targetCount / (sortedArray.length - 1))(sortedArray)
      .flatten
      .take(targetCount)

    if (sortMode == RAND) Random.shuffle(outputArray.toList).toArray
    else outputArray

  }

  /**
    * Method for generating ordinal Integer Data with Int.MinValue inserted to be replaced with null
    * @param targetCount Total number of Integers to generate in the series
    * @param start Starting point for the repeating series
    * @param step Distance between the repeating series values
    * @param mode sorting mode for the repeating arrays
    * @param distinctValues number of elements in the repeating series
    * @param targetNullRate Desired frequency of Int.MinValue to convert to null values
    * @param nullOffset adjustment to the starting position of Int.MinValue frequency
    * @return Array[Int] of Repeating Ordinal Integers with Int.MinValue for null replacement
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateRepeatingIntDataWithNulls(targetCount: Int,
                                        start: Int,
                                        step: Int,
                                        mode: String,
                                        distinctValues: Int,
                                        targetNullRate: Int,
                                        nullOffset: Int): Array[Int] = {
    generateRepeatingIntData(targetCount, start, step, mode, distinctValues).zipWithIndex
      .map {
        case (v, i) =>
          if ((i + nullOffset) % targetNullRate != 0.0) v else INT_FILL
      }
  }

  /**
    * Method for generating a series of Dates
    * @param targetCount Number of dates to generate
    * @param startingYear Starting year for the date series
    * @param startingMonth Starting month for the date series
    * @param startingDay Starting day for the date series
    * @return Array[String] Series of Dates
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateDates(targetCount: Int,
                    startingYear: Int,
                    startingMonth: Int,
                    startingDay: Int): Array[String] = {

    val start = new LocalDate(startingYear, startingMonth, startingDay)
    val dates = for (x <- 0 to targetCount) yield start.plusDays(x)
    dates.map(_.toString).toArray

  }

  /**
    * Method for generating a series of Dates with nulls inserted in the series
    * @param targetCount Number of dates to generate
    * @param startingYear Starting year for the date series
    * @param startingMonth Starting month for the date series
    * @param startingDay Starting day for the date series
    * @param targetNullRate Frequency with which to insert null values in the date column
    * @param nullOffset Adjustment to the starting position in the series to begin nulling data out
    * @return Array[String] series of dates with null values inserted
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateDatesWithNulls(targetCount: Int,
                             startingYear: Int,
                             startingMonth: Int,
                             startingDay: Int,
                             targetNullRate: Int,
                             nullOffset: Int): Array[String] = {
    generateDates(targetCount, startingYear, startingMonth, startingDay).zipWithIndex
      .map {
        case (v, i) => if ((i + nullOffset) % targetNullRate != 0.0) v else null
      }
  }

  /**
    * Method to generate an Array of Boolean values
    * @param targetCount Number of alternating Boolean values to generate
    * @return Array[Boolean]
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateBooleanData(targetCount: Int): Array[Boolean] = {
    Array.fill(targetCount)(Array(true, false)).flatten.take(targetCount)
  }

  /**
    * Method for generating Boolean data and forcing values to Null in the Array
    * @param targetCount Number of Boolean values to generate
    * @param targetNullRate Frequency of null values to insert
    * @param nullOffset Adjustment to the start position to insert null values
    * @return Array[Boolean] with null values inserted
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateBooleanDataWithNulls(targetCount: Int,
                                   targetNullRate: Int,
                                   nullOffset: Int): Array[Boolean] = {

    generateBooleanData(targetCount).zipWithIndex
      .map {
        case (v, i) => if ((i + nullOffset) % targetNullRate != 0.0) v else null
      }
      .map(_.asInstanceOf[Boolean])

  }

  /**
    * Method for generating a two-tailed distribution of data in a series of data
    * @param targetCount Number of elements to generate
    * @param start Starting point for the median of the series
    * @param step numeric distance between elements on a linear scale
    * @param mode sorting mode
    * @return Array of tailed Linear data
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateTailedData(targetCount: Int,
                         start: Double,
                         step: Double,
                         mode: String): Array[Double] = {

    val splitCount = math.ceil(targetCount / 2).toInt

    val mergedData = generateDoublesData(splitCount, start, step, mode).map(
      x => x * -1.0
    ) ++ generateDoublesData(splitCount, start, step, mode)
    val limitedData = mergedData.take(targetCount)
    getArrayMode(mode) match {
      case ASC  => limitedData.sortWith(_ < _)
      case DESC => limitedData.sortWith(_ > _)
      case RAND => Random.shuffle(limitedData.toList).toArray
    }

  }

  /**
    * Method for generating a two-tailed distribution of exponential data
    * @param targetCount Number of elements to generate
    * @param start starting point for the median of the series
    * @param step linear series distance between data points (which is then raised to the power provided)
    * @param mode sorting mode for the final series
    * @param power power to raise each element of the series by
    * @return Array of tailed exponential data
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateTailedExponentialData(targetCount: Int,
                                    start: Double,
                                    step: Double,
                                    mode: String,
                                    power: Int): Array[Double] = {
    val splitCount = math.ceil(targetCount / 2).toInt

    val mergedData = generateExponentialData(
      splitCount,
      start,
      step,
      mode,
      power
    ).map(x => x * -1) ++ generateExponentialData(
      splitCount,
      start,
      step,
      mode,
      power
    )
    val limitedData = mergedData.take(targetCount)
    getArrayMode(mode) match {
      case ASC  => limitedData.sortWith(_ < _)
      case DESC => limitedData.sortWith(_ > _)
      case RAND => Random.shuffle(limitedData.toList).toArray
    }
  }

  /**
    * Method for converting the temporary holders of 'null' values to actual nulls in a DataFrame
    * @param df DataFrame with temporary placeholder values
    * @return DataFrame with Nulls inserted
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def reassignToNulls(df: DataFrame): DataFrame = {

    val EXCLUSION_TYPES = Seq("boolean", "date", "time", "string")

    val namesAndTypes = df.schema
      .map(x => SchemaNamesTypes(x.name, x.dataType.typeName))
      .filterNot(x => EXCLUSION_TYPES.contains(x.dataType))

    namesAndTypes.foldLeft(df) { (df, x) =>
      {

        val dtype = x.dataType match {
          case "double"  => DOUBLE_FILL
          case "float"   => FLOAT_FILL
          case "integer" => INT_FILL
          case "long"    => LONG_FILL
        }
        df.withColumn(
          x.name,
          when(col(x.name) === dtype, null).otherwise(col(x.name))
        )
      }
    }
  }

  def generateStaticIntSeries(targetCount: Int, value: Int): Array[Int] = {
    Array.fill(targetCount)(value)
  }

  def generateStaticDoubleSeries(targetCount: Int,
                                 value: Double): Array[Double] = {
    Array.fill(targetCount)(value)
  }

  /**
    * Method for generating a classifier data set that has blocks of labels that will ensure significant separation
    * wihtin the feature vectors created. (Useful for testing items such as KSampling)
    * @param targetCount Number of elements to generate in each Array
    * @param start Starting value of the distinct values
    * @param step Distance between each of the distinct values
    * @param mode How to handle the associations of the blocks (ascending, descending, or random)
    * @param distinctValues Number of distinct elements to create within the Array
    * @return Array[Int] of data (in blocks or random)
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateIntegerBlocks(targetCount: Int,
                            start: Int,
                            step: Int,
                            mode: String,
                            distinctValues: Int): Array[Int] = {

    val sortMode = getArrayMode(mode)
    val blockGenSize =
      math.ceil(targetCount.toDouble / distinctValues.toDouble).toInt

    val calculatedStop = start + (distinctValues * step)

    sortMode match {
      case ASC =>
        (start to calculatedStop by step).toArray
          .flatMap(x => Array.fill(blockGenSize)(x))
      case DESC =>
        (start to calculatedStop by step).toArray.reverse
          .flatMap(x => Array.fill(blockGenSize)(x))
      case RAND =>
        Random
          .shuffle(
            (start to calculatedStop by step).toArray
              .flatMap(x => Array.fill(blockGenSize)(x))
              .toList
          )
          .toArray
    }

  }

  /**
    * Method for generating a skewed classification value for the label field to test out features such as KSampling and
    * stratified split
    * @param targetCount number of elements to generate in the Array
    * @param start Starting value
    * @param step distance between each block grouped value in the unique array
    * @param mode whether to sort of the blocks, ascending or descending, or to randomize them after generation
    * @param distinctValues number of unique values in the array
    * @return Array[Int] for classification label column values
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateIntegerBlocksSkewed(targetCount: Int,
                                  start: Int,
                                  step: Int,
                                  mode: String,
                                  distinctValues: Int): Array[Int] = {

    val sortMode = getArrayMode(mode)
    val blockGenSize =
      math.ceil(targetCount.toDouble / distinctValues.toDouble).toInt

    val calculatedStop = start + (distinctValues * step)

    var extractCount = targetCount

    val data = (0 until distinctValues).toArray.flatMap(x => {
      extractCount = extractCount / 2
      Array.fill(extractCount)(start + (x * step))
    })

    val remainder = targetCount - data.length

    sortMode match {
      case ASC =>
        Array.fill(remainder)(start) ++ data
      case DESC =>
        (Array.fill(remainder)(start) ++ data).reverse
      case RAND =>
        Random
          .shuffle(Array.fill(remainder)(start).toList ++ data.toList)
          .toArray
    }

  }

  /**
    * Method for generating a field of data with repeating blocks of Doubles
    * @param targetCount Number of elements to generate in each Array
    * @param start Starting value of the distinct values
    * @param step Distance between each of the distinct values
    * @param mode How to handle the associations of the blocks (ascending, descending, or random)
    * @param distinctValues Number of distinct elements to create within the Array
    * @return Array[Double] of data (in blocks or random)
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def generateDoublesBlocks(targetCount: Int,
                            start: Double,
                            step: Double,
                            mode: String,
                            distinctValues: Int): Array[Double] = {

    val sortMode = getArrayMode(mode)
    val blockGenSize =
      math.ceil(targetCount.toDouble / distinctValues.toDouble).toInt

    val calculatedStop = start + (distinctValues * step)

    sortMode match {
      case ASC =>
        (start to calculatedStop by step).toArray
          .flatMap(x => Array.fill(blockGenSize)(x))
      case DESC =>
        (start to calculatedStop by step).toArray.reverse
          .flatMap(x => Array.fill(blockGenSize)(x))
      case RAND =>
        Random
          .shuffle(
            (start to calculatedStop by step).toArray
              .flatMap(x => Array.fill(blockGenSize)(x))
              .toList
          )
          .toArray
    }

  }

}
