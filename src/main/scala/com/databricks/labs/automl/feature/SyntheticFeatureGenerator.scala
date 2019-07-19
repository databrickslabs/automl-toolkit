package com.databricks.labs.automl.feature

import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class SyntheticFeatureGenerator(data: DataFrame)
    extends SparkSessionWrapper
    with SyntheticFeatureBase {

  final val allowableLabelBalanceModes: List[String] =
    List("match", "percentage", "target")

  /**
    *
    * Available modes?
    *
    * 1. Bring all lesser labels to match largest label
    *   a. This needs to have a warning associated with it
    *   2. Bring all lesser labels to a % of the max label
    * 3. Override mode: Bring all labels to at least n
    *
    *
    *
    */
  private var _labelBalanceMode: String = defaultLabelBalanceMode
  private var _labelCol: String = defaultLabelCol
  private var _cardinalityThreshold: Int = defaultCardinalityThreshold
  private var _numericRatio: Double = defaultNumericRatio
  private var _numericTarget: Int = defaultNumericTarget

  def setLabelBalanceMode(value: String): this.type = {
    require(
      allowableLabelBalanceModes.contains(value),
      s"Label Balance Mode $value is not supported." +
        s"Must be one of: ${allowableLabelBalanceModes.mkString(", ")}"
    )
    _labelBalanceMode = value
    this
  }
  def setLabelCol(value: String): this.type = {
    _labelCol = value
    this
  }
  def setCardinalityThreshold(value: Int): this.type = {
    _cardinalityThreshold = value
    this
  }
  def setNumericRatio(value: Double): this.type = {
    require(
      value <= 1.0 & value > 0.0,
      s"Invalid Numeric Ratio entered!  Must be between 0 and 1." +
        s"${value.toString} is not valid."
    )
    _numericRatio = value
    this
  }
  def setNumericTarget(value: Int): this.type = {
    _numericTarget = value
    this
  }

  // TODO:
  // - pass in the DataFrame object, label column name, etc
  // - Apply setting for deciding what ratio of balance to generate
  // - Perform data checks
  // - Call the appropriate KSampling constructors
  // - Merge the synthetic DF back to the source of truth data set with an additional field denoting
  //  synthetic or not.

  // Need to create the Array[RowMapping] to pass into KSampling

  private def getMaxAndRest(
    full: Array[CardinalityPayload]
  ): (CardinalityPayload, Array[CardinalityPayload]) = {

    val sortedValues = full.sortWith(_.labelCounts > _.labelCounts)

    (sortedValues.head, sortedValues.drop(0))
  }

  /**
    * Private method for calculating the targets for all smaller classes for the percentage mode
    * @param max The most frequently occurring label
    * @param rest The remaining labels
    * @return Array[RowGenerationConfig] to supply the candidate target numbers for KSampling
    * @since 0.5.1
    * @author Ben Wilson
    */
  private def percentageTargets(
    max: CardinalityPayload,
    rest: Array[CardinalityPayload]
  ): Array[RowGenerationConfig] = {

    val targetValue = max.labelCounts

    rest
      .map { x =>
        val targetCounts = math
          .floor(targetValue * _numericRatio)
          .toInt - x.labelCounts

        if (targetCounts > 0) {

          RowGenerationConfig(x.labelValue, targetCounts)
        }
      }
      .asInstanceOf[Array[RowGenerationConfig]]
  }

  private def targetValidation(
    max: CardinalityPayload,
    rest: Array[CardinalityPayload]
  ): Array[RowGenerationConfig] = {

    rest
      .map { x =>
        if (_numericTarget > x.labelCounts) {
          RowGenerationConfig(x.labelValue, _numericTarget - x.labelCounts)
        }
      }
      .asInstanceOf[Array[RowGenerationConfig]]

  }

  private def matchValidation(
    max: CardinalityPayload,
    rest: Array[CardinalityPayload]
  ): Array[RowGenerationConfig] = {
    rest.map { x =>
      RowGenerationConfig(x.labelValue, max.labelCounts - x.labelCounts)
    }
  }

  private def determineRatios(): Array[RowGenerationConfig] = {

    val generatedGroups =
      LabelValidation(data, _labelCol, _cardinalityThreshold)

    val (max, rest) = getMaxAndRest(generatedGroups)

    _labelBalanceMode match {
      case "percentage" => percentageTargets(max, rest)
      case "target"     => targetValidation(max, rest)
      case "match"      => matchValidation(max, rest)
    }

  }

  //TODO: validation checks in case there are no elements to the collection!!
  // TODO: call into KSampling from a provided config and generate the synthetic data.

}

object SyntheticFeatureGenerator {
  def apply() = ???
}

case class CardinalityPayload(labelValue: Double, labelCounts: Int)

trait SyntheticFeatureBase {
  def defaultLabelCol: String = "label"
  def defaultCardinalityThreshold: Int = 20
  def defaultLabelBalanceMode: String = "percentage"
  def defaultNumericRatio: Double = 0.2
  def defaultNumericTarget: Int = 500
}

class LabelValidation(data: DataFrame) extends SyntheticFeatureBase {

  private var _labelCol: String = defaultLabelCol
  private var _cardinalityThreshold: Int = defaultCardinalityThreshold

  def setLabelCol(value: String): this.type = {
    _labelCol = value
    this
  }

  def setCardinalityThreshold(value: Int): this.type = {
    value match {
      case x if x > 20 =>
        println(
          s"[WARNING] setting value of cardinality threshold greater " +
            s"that 20 may indicate that this is a regression problem."
        )
    }
    _cardinalityThreshold = value
    this
  }

  /**
    * Private helper method for checking whether the provided DataFrame is within categorical
    * label type to ensure that there is not a 'runaway' condition of submitting
    * too many unique labels to generate data for.
    * @param grouped DataFrame: the grouped label data with counts.
    * @since 0.5.1
    * @author Ben Wilson
    */
  private def validateCardinalityCounts(grouped: DataFrame): Unit = {

    grouped.count() match {
      case x if x <= _cardinalityThreshold =>
        println(
          s"Unique counts of label " +
            s"column ${_labelCol} : ${x.toString}"
        )
      case _ =>
        throw new RuntimeException(
          s"[ALERT] Cardinality of label column is greater" +
            s"than threshold of "
        )
    }
  }

  /**
    * Private method for retrieving and validating the skew in the label column in order to support
    * KSampling synthetic label boosting.
    * @return Array[CardinalityPayload] of all of the counts of the labels throughout the data set.
    * @since 0.5.1
    * @author Ben Wilson
    */
  private def determineCardinality(): Array[CardinalityPayload] = {

    // Perform a DataFrame operation on the input label column
    val groupedLabel = data
      .select(col(_labelCol))
      .groupBy(col(_labelCol))
      .count()
      .as("counts")

    // Perform a validation check
    validateCardinalityCounts(groupedLabel)

    // Create the cardinality collection
    groupedLabel.collect.map { x =>
      CardinalityPayload(x.getAs[Double](_labelCol), x.getAs[Int]("counts"))
    }
  }

}

object LabelValidation {
  def apply(data: DataFrame,
            labelCol: String,
            cardinalityThreshold: Int): Array[CardinalityPayload] =
    new LabelValidation(data)
      .setLabelCol(labelCol)
      .setCardinalityThreshold(cardinalityThreshold)
      .determineCardinality()
}
