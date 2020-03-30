package com.databricks.labs.automl.exceptions

/**
  * @author Jas Bali
  * @since 0.6.1
  * This exception is thrown when there is a failure in the execution of a train pipeline
  *
  * @param pipelineId: Unique identifier for a pipeline run
  * @param cause: Root exception for pipeline execution failure
  */
final case class PipelineExecutionException(private val pipelineId: String, private val cause: Throwable)
  extends RuntimeException(s"Pipeline with ID $pipelineId failed", cause)
