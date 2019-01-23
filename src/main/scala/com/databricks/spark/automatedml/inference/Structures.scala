package com.databricks.spark.automatedml.inference

import org.apache.spark.sql.DataFrame

trait NAFill{
  def numericFillMap: Map[String, Double]
  def characterFillMap: Map[String, String]
}

abstract case class NAFillMaps() extends NAFill

abstract case class NAFillInference(
                                   labelAdjustment: Boolean,
                                   filledData: DataFrame
                                   ) extends NAFill

trait ConstructorReturn{
  def codeGen: String
  def data: DataFrame
}

abstract case class NAFillConstructorReturn(
                                  modelType: String
                                  ) extends ConstructorReturn

abstract case class VarianceFilterConstructorReturn(
                                          fieldsToDrop: Array[String]
                                          ) extends ConstructorReturn