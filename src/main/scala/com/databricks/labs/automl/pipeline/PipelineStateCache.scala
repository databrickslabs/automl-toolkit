package com.databricks.labs.automl.pipeline
import java.util.UUID

import org.apache.log4j.Logger

import scala.collection.mutable

/**
  * @author Jas Bali
  * A state cache for Pipeline context to maintain any internal state of a pipeline generation.
  * Handy when there is a need for dynamic runtime exchange of information required between Pipeline stages and/or
  * Pipeline context
  */
object PipelineStateCache {

  @transient lazy private val logger: Logger = Logger.getLogger(this.getClass)

  lazy private val pipelineStateCache = mutable.WeakHashMap[String, mutable.Map[String, Any]]()

  def addToPipelineCache(pipelineId: String, key: String, value: Any): Unit = {
    if(!pipelineStateCache.contains(pipelineId)) {
      pipelineStateCache += pipelineId -> mutable.Map.empty
    }
    pipelineStateCache += pipelineId -> (pipelineStateCache(pipelineId) += (key -> value))
    if(logger.isTraceEnabled) {
      logger.trace(
        s"""Added ($key, $value
           |) pair to Pipeline cache with ID: $pipelineId""".stripMargin)
    }
  }

  def getFromPipelineByIdAndKey(pipelineId: String, key: String): Any = {
    pipelineStateCache(pipelineId)(key)
  }

  def generatePipelineId(): String = UUID.randomUUID().toString

}
