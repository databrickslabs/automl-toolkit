package com.databricks.labs.automl.model.tools.split

import com.databricks.dbutils_v1.DBUtilsHolder.dbutils0
import com.databricks.labs.automl.model.tools.structures.{
  TrainSplitReferences,
  TrainTestPaths
}

object DeltaCacheCleanup {

  /**
    * Method for cleaning up all of the delta train/test paths that have been created during the modeling phase
    *
    * @param dataPayload Array of TrainSplitReferences containing the links to the delta paths
    * @since 0.7.1
    * @author Ben Wilson, Databricks
    */
  def removeCacheDirectories(dataPayload: Array[TrainSplitReferences]): Unit = {

    dataPayload.foreach { x =>
      {
        dbutils0.get().fs.rm(x.paths.train, true)
        dbutils0.get().fs.rm(x.paths.test, true)

      }
    }

  }

  /**
    * Internal method for cleaning up a kfold test/train data delta source
    *
    * @param dataPaths paths to test and train for a particular delta source
    * @since 0.7.1
    * @author Ben Wilson, Databricks
    */
  def removeTrainTestPair(dataPaths: TrainTestPaths): Unit = {

    dbutils0.get().fs.rm(dataPaths.train, true)
    dbutils0.get().fs.rm(dataPaths.test, true)

  }

}
