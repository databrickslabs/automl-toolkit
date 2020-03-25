package com.databricks.labs.automl.model.tools.split

import com.databricks.labs.automl.exploration.structures.FeatureImportanceConfig
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.MainConfig

object DataSplitCustodial {

  /**
    * Method for tidying up the cached, persisted, or delta-written test/train splits
    * @param splitData reference collection to the cached, persisted, or written-out delta tables
    * @param config main config, containing configuration references for how to handle the split data
    * @since 0.7.1
    * @author Ben Wilson, Databricks
    */
  def cleanCachedInstances(splitData: Array[TrainSplitReferences],
                           config: MainConfig): Unit = {

    splitData.foreach { x =>
      {
        config.geneticConfig.splitCachingStrategy match {
          case "cache" =>
            x.data.train.unpersist(true)
            x.data.test.unpersist(true)
          case "persist" =>
            x.data.train.unpersist()
            x.data.test.unpersist()
          case "delta" =>
            if (config.geneticConfig.deltaCacheBackingDirectoryRemovalFlag) {
              DeltaCacheCleanup.removeTrainTestPair(x.paths)
            }
        }
      }
    }

  }

  /**
    * Method for cleaning up the cached instances based on a FeatureImportance config object.
    * @param splitData reference collection to the cached, persisted, or written-out delta tables
    * @param config feature importances config, containing configuration references for how to handle the split data
    * @since 0.7.1
    * @author Ben Wilson, Databricks
    */
  def cleanCachedInstances(splitData: Array[TrainSplitReferences],
                           config: FeatureImportanceConfig): Unit = {

    splitData.foreach { x =>
      {
        config.splitCachingStrategy match {
          case "cache" =>
            x.data.train.unpersist(true)
            x.data.test.unpersist(true)
          case "persist" =>
            x.data.train.unpersist()
            x.data.test.unpersist()
          case "delta" =>
            if (config.deltaCacheBackingDirectoryRemovalFlag) {
              DeltaCacheCleanup.removeTrainTestPair(x.paths)
            }
        }
      }
    }

  }

}
