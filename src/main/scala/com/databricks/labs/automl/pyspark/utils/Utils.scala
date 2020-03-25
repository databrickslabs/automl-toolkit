package com.databricks.labs.automl.pyspark.utils

object Utils {

  def cleansNestedTypes(valuesMap: Map[String, Any]): Map[String, Any] = {
    val cleanMap: scala.collection.mutable.Map[String, Any] = scala.collection.mutable.Map()
    if (valuesMap.contains("fieldsToIgnoreInVector")) {
      cleanMap("fieldsToIgnoreInVector") = valuesMap("fieldsToIgnoreInVector").asInstanceOf[List[String]].toArray
    }
    if (valuesMap.contains("outlierFieldsToIgnore")) {
      cleanMap("outlierFieldsToIgnore") = valuesMap("outlierFieldsToIgnore").asInstanceOf[List[String]].toArray
    }
    if (valuesMap.contains("numericBoundaries")) {
      cleanMap("numericBoundaries") = valuesMap("numericBoundaries").asInstanceOf[Map[String, List[Any]]]
        .flatMap({ case (k, v) => {
          Map(k -> Tuple2(v.head.toString.toDouble, v(1).toString.toDouble))
        }})
    }
    cleanMap.toMap
  }

}
