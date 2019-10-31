package com.databricks.labs.automl.utils
import com.databricks.dbutils_v1.DBUtilsHolder.dbutils0
import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession

import scala.reflect.runtime.universe._

/**
  * Reflection Object brought to you courtesy of Jas Bali
  */
object DBUtilsHelper {

  private val logger: Logger = Logger.getLogger(this.getClass)

  private val ERROR_RETURN = "NA"

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

  private def wrapWithException(methodName: String): String = {
    try {
      if(!isLocalSparkSession) {
        return hijackProtectedMethods(methodName)
      }
    } catch {
      case e: Exception => {
        logger.debug(s"Method name $methodName not present on dbutils")
      }
    }
    ERROR_RETURN
  }

  /**
    * Gets the current running notebook path
    * @return
    */
  def getNotebookPath: String = {
    wrapWithException("notebookPath")
  }

  def getNotebookDirectory: String = {
    val notebookPath = getNotebookPath
    val notebookPathFinal = if(!notebookPath.equals(ERROR_RETURN)) notebookPath.substring(0, notebookPath.lastIndexOf("/")) + "/" else ERROR_RETURN
    notebookPathFinal
  }

  def getTrackingURI: String = {
    wrapWithException("apiUrl")
  }

  def getAPIToken: String = {
    wrapWithException("apiToken")
  }

  def isLocalSparkSession: Boolean = {
    SparkSession
      .builder()
      .getOrCreate()
      .sparkContext
      .isLocal
  }
}
