package com.databricks.labs.automl.exploration.analysis.common.structures

sealed trait ParamWrapper[+A] { def asOption: Option[A] }

case class Param[+A](value: A) extends ParamWrapper[A] {
  def asOption: Some[A] = Some(value)
}
case object NoParam extends ParamWrapper[Nothing] {
  def asOption: None.type = None
}

object ParamWrapper {
  implicit def valueToOption[A](x: A): ParamWrapper[A] = Param[A](x)
}
