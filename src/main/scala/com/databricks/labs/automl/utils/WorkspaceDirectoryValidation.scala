package com.databricks.labs.automl.utils

import scala.sys.process._

/**
  * Class for performing pre-check validation of mlflow working directories by interfacing (safely) with the
  * Workspace API.
  * Without performing these checks, on a sufficiently large and complex run, if the MlFlow logging project
  * directory does not yet exist in the Workspace, the job will fail to log to MlFlow.
  * Accessing the apply method on the object will:
  * 1. Check if the directory exists.  If it does, return Boolean true.
  * 2. If the path does not exist, attempt to make a mkdir POST to recursively create the pathing to the
  * target directory in the Workspace.
  * 3. Re-validate that the directory has been created and is set up correctly.  There is a linear back-off
  * sleep statement to make sure that there is a pause between requests to ensure that, if the REST service
  * is a bit overloaded, there is enough time to get the successful return confirmation of directory creation.
  * @param apiURL The shard URL
  * @param apiToken The user-specified token from the notebook context for authorization validation.
  */
class WorkspaceDirectoryValidation(apiURL: String,
                                   apiToken: String,
                                   path: String) {

  final private val statusAPI = s"$apiURL/api/2.0/workspace/get-status"
  final private val mkdirAPI = s"$apiURL/api/2.0/workspace/mkdirs"
  final private val header = s"Authentication: Bearer $apiToken"
  final private val baseCurl = Seq("curl", "-H", header, "-X")
  private val directoryMatch = "(\\/\\w+$)".r
  final private val adjustedPath = directoryMatch.replaceFirstIn(path, "")

  /**
    * Private method for generating the REST body statement for both requests.
    * @param adjPath String path in the Workspace for where to store the experimental results
    * @return The body statement
    */
  private def createPathBody(adjPath: String): String =
    s"""
       |{
       |  "path": "$adjPath"
       |}
     """.stripMargin

  /**
    * Private method for executing a recursive mkdir command to the Workspace
    * @param adjPath The path in the Workspace to create.
    * @return REST return statement (should be empty JSON)
    */
  private def createDir(adjPath: String): String = {
    val createCall = baseCurl ++ Seq(
      "POST",
      mkdirAPI,
      "-d",
      createPathBody(adjPath)
    )
    // Eat the stdout nonsense from the REST API call
    val buffer = new StringBuffer()
    createCall.lineStream_!(ProcessLogger(buffer append _)).toString()
  }

  /**
    * Helper method for performing a geometric-back-off sleep based on the effective retry policy.
    *
    * @example val waitTimes = (1 to 6).map(x => geomSleep(x, 1000))
    * waitTimes: scala.collection.immutable.IndexedSeq[Int] = Vector(1000, 1617, 3344, 6834, 13334, 24790)
    * @param counter the iteration of retry
    * @param pauseTime The amount of base wait time to apply for a back-off calculation.
    */
  private def geomSleep(counter: Int, pauseTime: Int): Unit = {
    val sleepTime = scala.math
      .ceil(pauseTime * scala.math.pow(counter, scala.math.log(counter)))
      .toInt
    Thread.sleep(sleepTime)
  }

  /**
    * Main method for checking whether the mlflow path exists to log run results to and if it does not,
    * attempts to create it as specified by the configuration.
    * @param cnt Loop counter (used in the recursive call)
    * @return Boolean: true if directory exists.
    */
  def validate(cnt: Int = 0): Boolean = {

    var attemptCounter = cnt

    val statusCall = baseCurl ++ Seq(
      "GET",
      statusAPI,
      "-d",
      createPathBody(adjustedPath)
    )

    val statusBuffer = new StringBuffer()
    val statusReturn =
      statusCall.lineStream_!(ProcessLogger(statusBuffer append _)).toString()

    val statusAnswer = try {
      statusReturn.split("\"")(1)
    } catch {
      case e: java.lang.ArrayIndexOutOfBoundsException =>
        println(
          s"The directory that you are attempting to log mlflow results to in your Workspace does not have " +
            s"the correct permissions for your account to create this directory.  Please provide a valid location " +
            s"in the Workspace.  Invalid access for path: $adjustedPath"
        )
        println(s"\n\n ${e.printStackTrace()}")
        throw e
    }

    statusAnswer match {
      case "error_code" =>
        attemptCounter += 1
        if (attemptCounter < 6) {
          createDir(adjustedPath)

          geomSleep(attemptCounter, 1000)

          validate(attemptCounter)

        } else {
          throw new RuntimeException(
            s"Unable to validate or create Workspace path to $adjustedPath. Ensure permissions" +
              s"are sufficient to have write access to Workspace Location.  " +
              s"\n\nSee: https://docs.databricks.com/user-guide/workspace.html for further information."
          )
        }
      case _ => true
    }
  }

}

object WorkspaceDirectoryValidation {

  def apply(apiURL: String, apiToken: String, path: String): Boolean =
    new WorkspaceDirectoryValidation(apiURL, apiToken, path).validate()

}
