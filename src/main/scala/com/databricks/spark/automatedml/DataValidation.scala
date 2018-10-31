package com.databricks.spark.automatedml


import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DataType, StructType}

import scala.collection.mutable.ListBuffer

trait DataValidation{
  def invalidateSelection(value: String, allowances: Seq[String]): String = {
    s"${allowances.foldLeft("")((a, b) => a + " " + b)}"
  }

  def extractSchema(schema: StructType): List[(DataType, String)] = {

    var preParsedFields = new ListBuffer[(DataType, String)]

    schema.map(x => preParsedFields += ((x.dataType, x.name)))

    preParsedFields.result
  }

  def extractTypes(data: DataFrame, labelColumn: String): (List[String], List[String]) = {

    val fieldExtraction = extractSchema(data.schema)
    var conversionFields = new ListBuffer[String]
    var vectorizableFields = new ListBuffer[String]

    fieldExtraction.map(x =>
      x._1.typeName match {
        case "string" => conversionFields += x._2
        case "integer" => vectorizableFields += x._2
        case "double" => vectorizableFields += x._2
        case "float" => vectorizableFields += x._2
        case "long" => vectorizableFields += x._2
        case "byte" => conversionFields += x._2
        case "boolean" => vectorizableFields += x._2
        case "binary" => vectorizableFields += x._2
        case _ => throw new UnsupportedOperationException(
          s"WARNING! Field ${x._2} is of type ${x._1} which is not supported!!")
      }
    )

    assert(vectorizableFields.contains(labelColumn),
      s"The provided Dataframe MUST contain a labeled column with the name '$labelColumn'")
    vectorizableFields -= labelColumn

    (vectorizableFields.result, conversionFields.result)

  }

}
