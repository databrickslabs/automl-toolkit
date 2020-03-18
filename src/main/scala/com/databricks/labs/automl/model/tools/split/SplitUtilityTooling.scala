package com.databricks.labs.automl.model.tools.split

import com.databricks.labs.automl.model.tools.structures.{
  TrainTestData,
  TrainTestPaths
}
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.sql.DataFrame

trait SplitUtilityTooling extends SparkSessionWrapper {

  def formRootPath(configStoreLocation: String): String = {

    configStoreLocation.takeRight(1) match {
      case "/" => configStoreLocation + "modeling_sources/"
      case _   => configStoreLocation + "/modeling_sources/"
    }

  }

  def formTrainTestPaths(configStoreLocation: String): TrainTestPaths = {

    val uniqueIdentifier = java.util.UUID.randomUUID()

    val rootPath = formRootPath(configStoreLocation)

    val trainPath = rootPath + s"train_$uniqueIdentifier"
    val testPath = rootPath + s"test_$uniqueIdentifier"

    TrainTestPaths(trainPath, testPath)

  }

  def storeLoadDelta(trainData: DataFrame,
                     testData: DataFrame,
                     paths: TrainTestPaths): TrainTestData = {

    // Write test data to delta location
    trainData.write.format("delta").save(paths.train)
    testData.write.format("delta").save(paths.test)

    // read from the location and provide a reference object to the reader

    TrainTestData(
      train = spark.read.format("delta").load(paths.train),
      test = spark.read.format("delta").load(paths.test)
    )

  }

}
