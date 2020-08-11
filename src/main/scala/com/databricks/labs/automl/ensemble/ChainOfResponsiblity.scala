package com.databricks.labs.automl.ensemble

import scala.collection.mutable.ArrayBuffer

private[ensemble] trait ChainOfResponsibilityTask {

}

private[ensemble] trait ChainOfResponsibilityExec{
  def run(task: ChainOfResponsibilityTask): ChainOfResponsibilityTask
}

private[ensemble] class ChainOfResponsibility {

  private final val tasks: ArrayBuffer[ChainOfResponsibilityExec] = ArrayBuffer[ChainOfResponsibilityExec]()

  def firstTask(task: ChainOfResponsibilityExec): Unit = {
    tasks += task
  }

  def addTask(task: ChainOfResponsibilityExec): Unit = {
    tasks += task
  }

  def lastTask(task: ChainOfResponsibilityExec): Unit = {
    tasks += task
  }

  def execChainOfResponsibility(): Unit = {

//    tasks.foldLeft()


  }


}
