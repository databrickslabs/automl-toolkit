package com.databricks.labs.automl.exploration.analysis.common.structures

private[analysis] sealed trait ParamWrapper[+A] { def asOption: Option[A] }

private[analysis] case class Param[+A](value: A) extends ParamWrapper[A] {
  def asOption: Some[A] = Some(value)
}

private[analysis] case object NoParam extends ParamWrapper[Nothing] {
  def asOption: None.type = None
}

private[analysis] object ParamWrapper {
  implicit def valueToOption[A](x: A): ParamWrapper[A] = Param[A](x)
}
