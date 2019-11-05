package com.databricks.labs.automl.utils.data

import org.apache.spark.sql.DataFrame

class CategoricalHandler(data: DataFrame, mode: String = "silent") {

  final val CATEGORICAL_MODES = List("silent", "warn")
  final val CARDINALITIES = List("approx", "exact")

  private var _cardinalityType: String = "exact"
  private var _precision = 0.05

  def setCardinalityType(value: String): this.type = {
    require(
      CARDINALITIES.contains(value),
      s"Specified cardinality type $value is not a member of ${CARDINALITIES
        .mkString(", ")}"
    )
    _cardinalityType = value
    this
  }

  def setPrecision(value: Double): this.type = {
    require(value >= 0.0, s"Precision must be greater than or equal to 0.")
    require(value <= 1.0, s"Precision must be less than or equal to 1.")
    _precision = value
    this
  }

  def validateCategoricalFields(fields: List[String],
                                cardinalityLimit: Int): Array[String] = {

    mode match {
      case "silent" =>
        FieldValidation.restrictFields(
          data,
          fields.toArray,
          _cardinalityType,
          cardinalityLimit.toLong,
          _precision
        )
      case _ =>
        FieldValidation.confirmCardinalityCheck(
          data,
          fields.toArray,
          _cardinalityType,
          cardinalityLimit.toLong,
          _precision
        )
    }

  }

}
