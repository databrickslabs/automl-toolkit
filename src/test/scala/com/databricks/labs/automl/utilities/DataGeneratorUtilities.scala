package com.databricks.labs.automl.utilities

import com.databricks.labs.automl.utils.structures.ArrayGeneratorMode

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

trait DataGeneratorUtilities {

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

  case class OutlierTestSchema(a: Double,
                               b: Double,
                               c: Double,
                               label: Int,
                               automl_internal_id: Long)

  case class NaFillTestSchema(dblData: Double,
                              fltData: Float,
                              intData: Int,
                              ordinalIntData: Int,
                              strData: String,
                              boolData: Boolean,
                              dateData: String,
                              label: Int,
                              automl_internal_id: Long)

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
  def generateStringArray(targetCount: Int,
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
  def generateStringArrayWithNulls(targetCount: Int,
                                   uniqueValueCount: Int,
                                   targetNullRate: Int,
                                   nullOffset: Int): Array[String] = {

    generateStringArray(targetCount, uniqueValueCount).zipWithIndex.map {
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
  def generateFibonacciArray(targetCount: Int): Array[Double] = {
    lazy val fibs
      : Stream[BigInt] = BigInt(0) #:: BigInt(1) #:: (fibs zip fibs.tail)
      .map(x => x._1 + x._2)
    fibs.take(targetCount).toArray.map(_.toDouble)
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
  def generateDoublesArray(targetCount: Int,
                           start: Double,
                           step: Double,
                           mode: String): Array[Double] = {

    val sortMode = getArrayMode(mode)

    val stoppingPoint = targetCount / step

    val doubleArray =
      Range.BigDecimal(start, stoppingPoint, step).toArray.map(_.toDouble)

    sortMode match {
      case ASC  => doubleArray.sortWith(_ < _)
      case DESC => doubleArray.sortWith(_ > _)
      case RAND => Random.shuffle(doubleArray.toList).toArray
    }
  }

  def generateDoublesArrayWithNulls(targetCount: Int,
                                    start: Double,
                                    step: Double,
                                    mode: String,
                                    targetNullRate: Int,
                                    nullOffset: Int): Array[Double] = {

    generateDoublesArray(targetCount, start, step, mode).zipWithIndex.map {
      case (v, i) =>
        if ((i + nullOffset) % targetNullRate != 0.0) v else Double.MinValue
    }

  }

  //TODO: generate log space distributions and exponential distributions based on a linear generated scale
//  def convertToLog(minScale: Double,
//                   maxScale: Double,
//                   value: Double): Double = {
//
//    val minVal = if (minScale == 0.0) 1.0E-10 else minScale
//
//    val b = log(maxScale / minVal) / (maxScale - minVal)
//
//    val a = maxScale / exp(b * maxScale)
//
//    a * exp(b * value)
//  }
//
//  def generateLogSpace(boundaries: NumericBoundaries,
//                       generatorCount: Int): Array[Double] = {
//
//    val space = new ArrayBuffer[Double]
//
//    val linearSpace = generateLinearSpace(boundaries, generatorCount)
//
//    linearSpace.foreach { x =>
//      space += convertToLog(boundaries.minimum, boundaries.maximum, x)
//    }
//
//    space.result.toArray
//
//  }

}
