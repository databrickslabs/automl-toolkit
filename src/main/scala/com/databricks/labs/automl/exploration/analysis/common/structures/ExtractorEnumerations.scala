package com.databricks.labs.automl.exploration.analysis.common.structures

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tree.{ContinuousSplit, InternalNode, Node, Split}

private[analysis] object NodeType extends Enumeration {
  type NodeType = Value
  val NODE, LEAF = Value
}

private[analysis] object SplitType extends Enumeration {
  type SplitType = Value
  val CONTINUOUS, CATEGORICAL = Value
}

private[analysis] object NodeDetermination {

  import NodeType._
  import SplitType._

  def nodeType(node: Node): NodeType = node match {
    case _: InternalNode => NODE
    case _               => LEAF
  }

  def splitType(split: Split): SplitType = split match {
    case _: ContinuousSplit => CONTINUOUS
    case _                  => CATEGORICAL
  }

}

private[analysis] object PayloadType extends Enumeration {
  type PayloadType = Value
  val MODEL, PIPELINE = Value
}

private[analysis] object PayloadDetermination {

  import PayloadType._

  def payloadType[T](value: T): PayloadType = {
    value match {
      case _: PipelineModel => PIPELINE
      case _                => MODEL
    }
  }

}
