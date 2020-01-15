package com.databricks.labs.automl.feature.tools

import com.databricks.labs.automl.feature.SyntheticFeatureBase
import com.databricks.labs.automl.feature.structures.CardinalityPayload
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

class LabelValidation(data: DataFrame) extends SyntheticFeatureBase {

  private var _cardinalityThreshold: Int = defaultCardinalityThreshold

  def setCardinalityThreshold(value: Int): this.type = {
    value match {
      case x if x > 20 =>
        println(
          s"[WARNING] setting value of cardinality threshold greater " +
            s"that 20 may indicate that this is a regression problem."
        )
      case _ => None
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
    * @throws RuntimeException() If the cardinality of the label column exceeds the thresholds
    */
  @throws(classOf[RuntimeException])
  private def validateCardinalityCounts(grouped: DataFrame): Unit = {

    val logger: Logger = Logger.getLogger(this.getClass)

    grouped.count() match {
      case x if x <= _cardinalityThreshold =>
        logger.log(
          Level.INFO,
          s"Unique counts of label " +
            s"column ${_labelCol} : ${x.toString}"
        )
      case _ =>
        throw new RuntimeException(
          s"[ALERT] Cardinality of label column is greater" +
            s"than threshold of: ${_cardinalityThreshold.toString}"
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

    // Perform a validation check
    validateCardinalityCounts(groupedLabel)

    // Get the data type of the label column
    val labelType =
      data.schema.filter(x => x.name == _labelCol).head.dataType.typeName

    // Create the cardinality collection
    groupedLabel.collect.map { x =>
      labelType match {
        case "double" =>
          CardinalityPayload(
            x.getAs[Double](_labelCol),
            x.getAs[Long]("count").toInt
          )
        case "integer" =>
          CardinalityPayload(
            x.getAs[Int](_labelCol).toDouble,
            x.getAs[Long]("count").toInt
          )
        case "float" =>
          CardinalityPayload(
            x.getAs[Float](_labelCol).toDouble,
            x.getAs[Long]("count").toInt
          )
        case _ =>
          throw new RuntimeException(
            s"The data type of the label column ${_labelCol} is: $labelType" +
              s"which is not supported.  Must be one of: DoubleType, IntegerType, or FloatType"
          )
      }
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
