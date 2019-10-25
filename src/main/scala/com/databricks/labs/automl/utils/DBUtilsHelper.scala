package com.databricks.labs.automl.utils
import com.databricks.dbutils_v1.DBUtilsHolder.dbutils0
import org.apache.spark.sql.SparkSession

import scala.reflect.runtime.universe._

/**
  * Reflection Object brought to you courtesy of Jas Bali
  */
object DBUtilsHelper {
  protected def reflectedDBUtilsMethod(methodName: String): Array[String] = {
    Array(
      dbutils0
        .get()
        .notebook
        .getContext()
        .getClass
        .getMethods
        .map(_.getName)
        .filter(_.equals(methodName))
    ).head
  }
  protected def hijackProtectedMethods(methodName: String): String = {
    val ctx = dbutils0.get().notebook.getContext()
    val mirrorContext = runtimeMirror(getClass.getClassLoader)
      .reflect(dbutils0.get().notebook.getContext())
    val result = reflectedDBUtilsMethod(methodName)
      .map(x => {
        scala.reflect.runtime.universe
          .typeOf[ctx.type]
          .decl(TermName(x))
          .asMethod
      })
      .map(mirrorContext.reflectMethod(_).apply())
    result(0).asInstanceOf[Option[_]].get.toString
  }

  /**
    * Gets the current running notebook path
    * @return
    */
  def getNotebookPath: String = {
    if(!isLocalSparkSession) {
      return hijackProtectedMethods("notebookPath")
    }
    "NA"
  }
  def getNotebookDirectory: String = {
    if(!isLocalSparkSession) {
      val notebookPath = getNotebookPath
      return notebookPath.substring(0, notebookPath.lastIndexOf("/")) + "/"
    }
    "NA"
  }
  def getTrackingURI: String = {
    if(!isLocalSparkSession) {
      return hijackProtectedMethods("apiUrl")
    }
    "NA"
  }
  def getAPIToken: String = {
    if(!isLocalSparkSession) {
      return hijackProtectedMethods("apiToken")
    }
    "NA"
  }

  def isLocalSparkSession: Boolean = {
    SparkSession
      .builder()
      .getOrCreate()
      .sparkContext
      .isLocal
  }
}
