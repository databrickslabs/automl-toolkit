package com.databricks.labs.automl.model.tools.split

import com.databricks.labs.automl.model.tools.structures.{
  TrainSplitReferences,
  TrainTestData,
  TrainTestPaths
}
import org.apache.spark.sql.DataFrame

/**
  * Train / Test split handler class
  * @param mainDataset Dataset that contains feature vector, out of DataPrep phase, ready to be split into
  * @param kIterations number of 'copies' of the split to perform in order to fulfill the number of kFold models to be built
  * @param splitMethod The type of split being performed (i.e. 'stratified', 'random', 'kSample')
  * @param labelColumn Name of the label column
  * @param rootDir Source directory to use to build the delta persisted data sets if using 'delta' mode in persistMode
  * @param persistMode 'cache', 'persist' or 'delta' - how to retain each of the kFold train/test splits.
  * @param modelFamily The model family in order to determine how many parts in which to repartition the train and test
  *                    splits for optimal performance.
  * @since 0.7.1
  * @author Ben Wilson, Databricks
  */
class DataSplitUtility(mainDataset: DataFrame,
                       kIterations: Int,
                       splitMethod: String,
                       labelColumn: String,
                       rootDir: String,
                       persistMode: String,
                       modelFamily: String)
    extends SplitUtilityTooling {

  final val uniqueLabels = mainDataset.select(labelColumn).distinct().collect()

  /**
    * Method for persisting the train test splits to local disk.
    * @return Array[TrainSplitReferences], containing pointers to the Data, organized by kFold index, as well as
    *         dummy values for pathing.
    * @since 0.7.1
    * @author Ben Wilson, Databricks
    */
  private def trainSplitPersist: Array[TrainSplitReferences] = {

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

  /**
    * Method for caching the train test splits in memory.
    * @return Array[TrainSplitReferences], containing pointers to the Data, organized by kFold index, as well as
    *         dummy values for pathing.
    * @since 0.7.1
    * @author Ben Wilson, Databricks
    */
  private def trainSplitCache: Array[TrainSplitReferences] = {

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

  /**
    * Method for writing the train test splits to dbfs as a delta data source
    *
    * @return Array[TrainSplitReferences], containing pointers to the Data as stored by Delta, organized by kFold index,
    *         as well as the values for pathing for eventual cleanup.
    * @since 0.7.1
    * @author Ben Wilson, Databricks
    */
  private def trainSplitDelta: Array[TrainSplitReferences] = {

    (0 to kIterations).map { x =>
      val Array(train, test) =
        genTestTrain(mainDataset, scala.util.Random.nextLong(), uniqueLabels)

      val deltaPaths = formTrainTestPaths(rootDir)

      val deltaReferences = storeLoadDelta(train, test, deltaPaths)

      TrainSplitReferences(x, deltaReferences, deltaPaths)
    }.toArray

  }

  /**
    * Wrapper interface for performing the splits, dependent on mode
    * @return Array[TrainSplitReferences] from the above methods.
    */
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
