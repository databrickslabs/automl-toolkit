package com.databricks.labs.automl.exploration.analysis.common.structures

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tree.{ContinuousSplit, InternalNode, Node, Split}

object NodeType extends Enumeration {
  type NodeType = Value
  val NODE, LEAF = Value
}

object SplitType extends Enumeration {
  type SplitType = Value
  val CONTINUOUS, CATEGORICAL = Value
}

object NodeDetermination {

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

object PayloadType extends Enumeration {
  type PayloadType = Value
  val MODEL, PIPELINE = Value
}

object PayloadDetermination {

  import PayloadType._

  def payloadType[T](value: T): PayloadType = {
    value match {
      case _: PipelineModel => PIPELINE
      case _                => MODEL
    }
  }

}
