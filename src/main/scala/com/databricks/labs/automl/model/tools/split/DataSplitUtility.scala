package com.databricks.labs.automl.model.tools.split

import com.databricks.labs.automl.model.tools.structures.{
  TrainSplitReferences,
  TrainTestData,
  TrainTestPaths
}
import org.apache.spark.sql.DataFrame

class DataSplitUtility(mainDataset: DataFrame,
                       kIterations: Int,
                       splitMethod: String,
                       labelColumn: String,
                       rootDir: String,
                       persistMode: String,
                       modelFamily: String)
    extends SplitUtilityTooling {

  final val uniqueLabels = mainDataset.select(labelColumn).distinct().collect()

    def trainSplitPersist: Array[TrainSplitReferences] = {

    val optimalParts = modelFamily match {
      case "XGBoost" => xgbWorkers
      case _         => optimalJVMModelPartitions
    }

    (0 to kIterations).map { x =>
      val Array(train, test) =
        genTestTrain(mainDataset, scala.util.Random.nextLong(), uniqueLabels)
      val (persistedTrain, persistedTest) =
        optimizeTestTrain(train, test, optimalParts, shuffle = true)

      TrainSplitReferences(
        x,
        TrainTestData(persistedTrain, persistedTest),
        TrainTestPaths("", "")
      )

    }.toArray

  }

  def trainSplitCache: Array[TrainSplitReferences] = {

    val optimalParts = modelFamily match {
      case "XGBoost" => xgbWorkers
      case _         => optimalJVMModelPartitions
    }

    (0 to kIterations).map { x =>
      val Array(train, test) =
        genTestTrain(mainDataset, scala.util.Random.nextLong(), uniqueLabels)

      val trainCache = train.repartition(optimalParts).cache()
      val testCache = test.repartition(optimalParts).cache()

      trainCache.foreach(_ => ())
      testCache.foreach(_ => ())

      TrainSplitReferences(
        x,
        TrainTestData(trainCache, testCache),
        TrainTestPaths("", "")
      )
    }.toArray

  }

  def trainSplitDelta: Array[TrainSplitReferences] = {

    (0 to kIterations).map { x =>
      val Array(train, test) =
        genTestTrain(mainDataset, scala.util.Random.nextLong(), uniqueLabels)

      val deltaPaths = formTrainTestPaths(rootDir)

      val deltaReferences = storeLoadDelta(train, test, deltaPaths)

      TrainSplitReferences(x, deltaReferences, deltaPaths)
    }.toArray

  }

  def performSplit: Array[TrainSplitReferences] = {

    persistMode match {
      case "persist" => trainSplitPersist
      case "delta"   => trainSplitDelta
      case "cache"   => trainSplitCache
      case _ =>
        throw new UnsupportedOperationException(
          s"Train Split mode $persistMode is not supported."
        )
    }

  }

}

object DataSplitUtility {

  def split(mainDataSet: DataFrame,
            kIterations: Int,
            splitMethod: String,
            labelColumn: String,
            rootDir: String,
            persistMode: String,
            modelFamily: String): Array[TrainSplitReferences] =
    new DataSplitUtility(
      mainDataSet,
      kIterations,
      splitMethod,
      labelColumn,
      rootDir,
      persistMode,
      modelFamily
    ).performSplit

}
