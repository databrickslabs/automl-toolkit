package com.databricks.labs.automl.model.tools.split

import com.databricks.labs.automl.model.tools.structures.{
  TrainSplitReferences,
  TrainTestData,
  TrainTestPaths
}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame

class DataSplitUtility(mainDataset: DataFrame,
                       kIterations: Int,
                       splitMethod: String,
                       labelColumn: String,
                       rootDir: String,
                       persistMode: String,
                       modelFamily: String,
                       parallelism: Int,
                       trainPortion: Double,
                       syntheticCol: String,
                       trainSplitChronologicalColumn: String,
                       trainSplitChronologicalRandomPercentage: Double,
                       reductionFactor: Double)
    extends SplitUtilityTooling {

  @transient private val logger: Logger = Logger.getLogger(this.getClass)

  final val uniqueLabels = mainDataset.select(labelColumn).distinct().collect()

  def trainSplitPersist: Array[TrainSplitReferences] = {

    val optimalParts = modelFamily match {
      case "XGBoost" => PerformanceSettings.xgbWorkers(parallelism)
      case _         => PerformanceSettings.optimalJVMModelPartitions(parallelism)
    }

    (0 until kIterations).map { x =>
      val Array(train, test) =
        SplitOperators.genTestTrain(
          mainDataset,
          scala.util.Random.nextLong(),
          uniqueLabels,
          splitMethod,
          labelColumn,
          trainPortion,
          syntheticCol,
          trainSplitChronologicalColumn,
          trainSplitChronologicalRandomPercentage,
          reductionFactor
        )
      logger.log(
        Level.DEBUG,
        s"DEBUG: Generated train/test split for kfold $x. Beginning persist."
      )
      val (persistedTrain, persistedTest) =
        SplitOperators.optimizeTestTrain(
          train,
          test,
          optimalParts,
          shuffle = true
        )

      TrainSplitReferences(
        x,
        TrainTestData(persistedTrain, persistedTest),
        TrainTestPaths("", "")
      )

    }.toArray

  }

  def trainSplitCache: Array[TrainSplitReferences] = {

    val optimalParts = modelFamily match {
      case "XGBoost" => PerformanceSettings.xgbWorkers(parallelism)
      case "RandomForest" =>
        PerformanceSettings.optimalJVMModelPartitions(parallelism) * 4
      case _ => PerformanceSettings.optimalJVMModelPartitions(parallelism)
    }

    (0 to kIterations).map { x =>
      val Array(train, test) =
        SplitOperators.genTestTrain(
          mainDataset,
          scala.util.Random.nextLong(),
          uniqueLabels,
          splitMethod,
          labelColumn,
          trainPortion,
          syntheticCol,
          trainSplitChronologicalColumn,
          trainSplitChronologicalRandomPercentage,
          reductionFactor
        )

      logger.log(
        Level.DEBUG,
        s"DEBUG: Generated train/test split for kfold $x. Beginning cache to memory."
      )

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
        SplitOperators.genTestTrain(
          mainDataset,
          scala.util.Random.nextLong(),
          uniqueLabels,
          splitMethod,
          labelColumn,
          trainPortion,
          syntheticCol,
          trainSplitChronologicalColumn,
          trainSplitChronologicalRandomPercentage,
          reductionFactor
        )

      val deltaPaths = formTrainTestPaths(rootDir)

      val deltaReferences = storeLoadDelta(train, test, deltaPaths)

      logger.log(
        Level.DEBUG,
        s"DEBUG: Generated train/test split for kfold $x. Stored tables to Delta paths."
      )

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
            modelFamily: String,
            parallelism: Int,
            trainPortion: Double,
            syntheticCol: String,
            trainSplitChronologicalColumn: String,
            trainSplitChronologicalRandomPercentage: Double,
            reductionFactor: Double): Array[TrainSplitReferences] =
    new DataSplitUtility(
      mainDataSet,
      kIterations,
      splitMethod,
      labelColumn,
      rootDir,
      persistMode,
      modelFamily,
      parallelism,
      trainPortion,
      syntheticCol,
      trainSplitChronologicalColumn,
      trainSplitChronologicalRandomPercentage,
      reductionFactor
    ).performSplit

}
