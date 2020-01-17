package com.databricks.labs.automl.utils.structures

import org.apache.spark.sql.types.DataType

case class FieldTypes(numericFields: List[String],
                      categoricalFields: List[String],
                      dateFields: List[String],
                      timeFields: List[String],
                      booleanFields: List[String])

case class FieldDefinitions(dataType: DataType, fieldName: String)

case class FieldPairs(left: String, right: String)

case class FieldCorrelationPayload(primaryColumn: String,
                                   pairs: FieldPairs,
                                   correlation: Double)

case class FieldCorrelationAggregationStats(rowCounts: Double,
                                            averageMap: Map[String, Double])

case class FieldRemovalPayload(dropFields: Array[String],
                               retainFields: Array[String])
