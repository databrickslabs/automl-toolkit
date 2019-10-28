package com.databricks.labs.automl.pipeline
import java.util.UUID

import scala.collection.mutable

/**
  * @author Jas Bali
  * A state cache for Pipeline context to maintain any internal state of a pipeline generation.
  * Handy when there is a need for dynamic runtime exchange of information required between Pipeline stages and/or
  * Pipeline context
  */
object PipelineStateCache {

  lazy private val pipelineStateCache = mutable.WeakHashMap[String, mutable.Map[String, Any]]()

  def addToPipelineCache(pipelineId: String, key: String, value: Any): Unit = {
    if(!pipelineStateCache.contains(pipelineId)) {
      pipelineStateCache += pipelineId -> mutable.Map.empty
    }
    pipelineStateCache += pipelineId -> (pipelineStateCache(pipelineId) += (key -> value))
  }

  def getFromPipelineByIdAndKey(pipelineId: String, key: String): Any = {
    pipelineStateCache(pipelineId)(key)
  }

  def generatePipelineId(): String = UUID.randomUUID().toString

}
