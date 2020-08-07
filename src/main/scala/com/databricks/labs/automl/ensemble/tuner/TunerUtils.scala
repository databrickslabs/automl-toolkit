package com.databricks.labs.automl.ensemble.tuner

import com.databricks.labs.automl.model.tools.split.{DataSplitCustodial, DataSplitUtility}
import com.databricks.labs.automl.model.tools.structures.TrainSplitReferences
import com.databricks.labs.automl.params.{DataGeneration, MainConfig}
import org.apache.spark.storage.StorageLevel

object TunerUtils {

  def cacheAndBuildSplitTrainTestData(mainConfig: MainConfig,
                              payload: DataGeneration): Array[TrainSplitReferences] =  {
    val cachedData = if (mainConfig.dataPrepCachingFlag) {
    val data = payload.data.persist(StorageLevel.MEMORY_AND_DISK)
    data.foreach(_ => ())
    data
  } else {
    payload.data
  }

    DataSplitUtility.split(
    cachedData,
      mainConfig.geneticConfig.kFold,
      mainConfig.geneticConfig.trainSplitMethod,
      mainConfig.labelCol,
      mainConfig.geneticConfig.deltaCacheBackingDirectory,
      mainConfig.geneticConfig.splitCachingStrategy,
      mainConfig.modelFamily,
      mainConfig.geneticConfig.parallelism,
      mainConfig.geneticConfig.trainPortion,
      mainConfig.geneticConfig.kSampleConfig.syntheticCol,
      mainConfig.geneticConfig.trainSplitChronologicalColumn,
      mainConfig.geneticConfig.trainSplitChronologicalRandomPercentage,
      mainConfig.dataReductionFactor
    )
  }

  def cleanTrainSplitCache(mainConfig: MainConfig, splitData: Array[TrainSplitReferences]): Unit = {
    DataSplitCustodial.cleanCachedInstances(splitData, mainConfig)
  }

}
