package com.databricks.spark.automatedml.utils

trait AutomationTools {

  case class GenericModelReturn(
                                 hyperParams: Map[String, Any],
                                 model: Any,
                                 score: Double,
                                 metrics: Map[String, Double],
                                 generation: Int
                               )

  def extractPayload(cc: Product): Map[String, Any] = {
    val values = cc.productIterator
    cc.getClass.getDeclaredFields.map{
      _.getName -> (values.next() match {
        case p: Product if p.productArity > 0 => extractPayload(p)
        case x => x
      })
    }.toMap
  }

}