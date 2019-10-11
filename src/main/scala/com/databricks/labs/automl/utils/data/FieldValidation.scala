package com.databricks.labs.automl.utils.data

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

class FieldValidation(data: DataFrame) {

  final val CARDINALITIES: Array[String] = Array("approx", "exact")

  final val schema: StructType = data.schema
  final val fieldNames: Array[String] = schema.names

  /**
    * Private method for validating the presence of fields in the DataFrame's schema
    *
    * @param fields Array[String] a list of the fields to validate against the DataFrame's schema
    * @throws IllegalArgumentException if the fields to be tested are not in the DataFrame's schema
    * @author Ben Wilson
    * @since 0.5.2
    */
  @throws(classOf[IllegalArgumentException])
  private def validateMembership(fields: Array[String]): Unit = {

    fields.foreach(
      x =>
        require(
          fieldNames.contains(x),
          s"ERROR! Schema does not contain field $x"
      )
    )
  }

  /**
    * Private method for determining the cardinality for a particular Array of fields
    *
    * @param fields Array of field names to validate their cardinality before executing a potentially expensive
    *               operation.
    * @param cardinalityType The type of distinct to use for each column.  [Either "approx or "exact"]
    * @param cardinalityLimit The limit, above which, an exception will be thrown for attempting to send
    *                         a DataFrame with categorical data that exceeds the desired threshold for inclusion in a
    *                         feature vector.
    * @param precision For approx distinct, the precision by which the the approximate distinct will be performed.
    * @throws AssertionError if the cardinality is too high for a field
    * @author Ben Wilson
    * @since 0.5.2
    */
  @throws(classOf[AssertionError])
  private def checkCardinality(fields: Array[String],
                               cardinalityType: String,
                               cardinalityLimit: Long,
                               precision: Double = 0.05): Unit = {

    fields.foreach { x =>
      val cardinality =
        calculateCardinality(data, x, cardinalityType, precision).rdd
          .map(r => r.getLong(0))
          .take(1)(0)
      assert(
        cardinality <= cardinalityLimit,
        s"Field $x has a cardinality of $cardinality which exceeds the " +
          s"limit of: $cardinalityLimit"
      )
    }

  }

  /**
    * Private method for switching between the cardinality methodologies (either exact or approximate)
    *
    * @param df The Dataframe for which a cardinality will be applied for a particular field
    * @param field The field to calculate the cardinality for
    * @param cardinalityType The type of cardinality check to use [either approx or exact]
    * @param precision The precision for an approx distinct check
    * @return The Dataframe (1 row) that contains the cardinality distinct check
    * @author Ben Wilson
    * @since 0.5.2
    */
  private def calculateCardinality(df: DataFrame,
                                   field: String,
                                   cardinalityType: String,
                                   precision: Double = 0.05): DataFrame = {

    cardinalityType match {
      case "exact" => df.select(countDistinct(field))
      case _       => df.select(approx_count_distinct(field, rsd = precision))
    }

  }

  /**
    * Validation method for ensuring that the fields specified have a cardinality below a set threshold
    *
    * @param fields Fields to test as an Array of Column Names
    * @param cardinalityType The type of distinct check to perform to calculate the cardinality
    *                        [either 'exact' or 'approx']
    * @param cardinalityLimit The limit, above which, the check will fail.
    * @throws AssertionError if the cardinality of a field exceeds the threshold
    * @author Ben Wilson
    * @since 0.5.2
    */
  @throws(classOf[AssertionError])
  def validateCardinality(fields: Array[String],
                          cardinalityType: String,
                          cardinalityLimit: Long,
                          precision: Double = 0.05): Array[String] = {

    validateMembership(fields)
    checkCardinality(fields, cardinalityType, cardinalityLimit, precision)
    fields
  }

  /**
    * Method for filtering out any fields that are above a certain cardinality threshold to protect against
    * creating unmanageably large feature vectors or computationally extreme StringIndexed values
    *
    * @param fields Fields to validate cardinality for
    * @param cardinalityType The mode of cardinality checking [either "approx" for approximate distinct or "exact"]
    * @param cardinalityLimit The limitation above which any field's cardinality will cause the field to be culled
    *                         from the collection of fields to perform an operation on
    * @param precision The precision set point for approx_distinct calculations for expected high cardinality fields
    *                  or large data sets.
    * @return Array[String] of column names whose cardinality is below the threshold specified by cardinalityLimit
    * @author Ben Wilson
    * @since 0.5.2
    */
  def restrictFieldsBasedOnCardinality(
    fields: Array[String],
    cardinalityType: String,
    cardinalityLimit: Long,
    precision: Double = 0.05
  ): Array[String] = {

    validateMembership(fields)

    fields
      .map { x =>
        val cardinality =
          calculateCardinality(data, x, cardinalityType, precision).rdd
            .map(r => r.getAs[Long](0))
            .take(1)(0)
        cardinality match {
          case y if y <= cardinalityLimit => x
          case _ => ""
        }
      }
      .filterNot(x => x.equals(""))
  }
}

/**
  * Companion Object
  */
object FieldValidation {

  def apply(data: DataFrame): FieldValidation = new FieldValidation(data)

  def confirmCardinalityCheck(data: DataFrame,
                              fields: Array[String],
                              cardinalityType: String,
                              cardinalityLimit: Long,
                              precision: Double = 0.05): Array[String] =
    this
      .apply(data)
      .validateCardinality(fields, cardinalityType, cardinalityLimit, precision)

  def restrictFields(data: DataFrame,
                     fields: Array[String],
                     cardinalityType: String,
                     cardinalityLimit: Long,
                     precision: Double = 0.05): Array[String] =
    this
      .apply(data)
      .restrictFieldsBasedOnCardinality(
        fields,
        cardinalityType,
        cardinalityLimit,
        precision
      )

}
