package com.databricks.labs.automl.exceptions

import java.util.concurrent.{ArrayBlockingQueue, ThreadPoolExecutor, TimeUnit}

import scala.concurrent.{ExecutionContext, ExecutionContextExecutor}

/**
  * @author Jas Bali
  * Provides thread pools by size and can be used with [[ExecutionContextExecutor]] to ensure a true Thread Pool
  * is created, so that threads are reused and limited
  */
object ThreadPoolsBySize {

  private lazy val SMALL_RUNNING_TASKS_TP_CORE_SIZE = 2
  private lazy val SMALL_RUNNING_TASKS_TP_INITIAL_MAX_SIZE = 20

  private lazy val SMALL_RUNNING_TASKS_TP = new ThreadPoolExecutor(
    SMALL_RUNNING_TASKS_TP_CORE_SIZE, SMALL_RUNNING_TASKS_TP_INITIAL_MAX_SIZE,
    15, TimeUnit.MINUTES, new ArrayBlockingQueue[Runnable](100))

  lazy val SMALL_RUNNING_TASKS_TP_EC: ExecutionContextExecutor = ExecutionContext.fromExecutor(SMALL_RUNNING_TASKS_TP)

  // Add more thread pools as needed

  def withScalaExecutionContext(parallelism: Int = SMALL_RUNNING_TASKS_TP_INITIAL_MAX_SIZE): ExecutionContextExecutor = {
    if(parallelism > SMALL_RUNNING_TASKS_TP_INITIAL_MAX_SIZE ) {
      SMALL_RUNNING_TASKS_TP.setMaximumPoolSize(parallelism)
    }
   ExecutionContext.fromExecutor(SMALL_RUNNING_TASKS_TP)
  }
}
