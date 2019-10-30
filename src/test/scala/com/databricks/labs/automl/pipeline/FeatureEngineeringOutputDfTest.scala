package com.databricks.labs.automl.pipeline

import java.sql.{Date, Timestamp}

import com.databricks.labs.automl.AbstractUnitSpec
import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.rand
import org.apache.spark.sql.types._

import scala.collection.JavaConverters._

class FeatureEngineeringOutputDfTest extends AbstractUnitSpec {

  "it" should "do this" in {
    val random = new scala.util.Random
    // Create a start and end value pair
    val startDay = 1
    val endDay = 24
    val startMonth = 1
    val endMonth = 12
    val startYear = 2000
    val endYear = 2020

    val dateColToBeTransformed = "download_date"
    val tsColToBeTransformed = "event_ts"
    val sourceDFTmp = SparkSession
      .builder().
      master("local[*]")
      .appName("providentiaml-unit-tests")
      .getOrCreate().createDataFrame(
      (List.fill(50)(Row(
        scala.math.abs(scala.util.Random.nextLong()),
        scala.math.abs(scala.util.Random.nextLong()) + "",
        Date.valueOf(s"${startYear + random.nextInt( (endYear - startYear) + 1 )}-${startMonth + random.nextInt( (endMonth - startMonth) + 1 )}-${startDay + random.nextInt( (endDay - startDay) + 1 )}"),
        Timestamp.valueOf(s"${startYear + random.nextInt( (endYear - startYear) + 1 )}-${startMonth + random.nextInt( (endMonth - startMonth) + 1 )}-${startDay + random.nextInt( (endDay - startDay) + 1 )} ${startMonth + random.nextInt( (endMonth - startMonth) + 1 )}:${startMonth + random.nextInt( (endMonth - startMonth) + 1 )}:${startMonth + random.nextInt( (endMonth - startMonth) + 1 )}.${startMonth + random.nextInt( (endMonth - startMonth) + 1 )}"),
        "pass"
      )) ++ List.fill(50)(Row(
        scala.math.abs(scala.util.Random.nextLong()),
        scala.math.abs(scala.util.Random.nextLong()) + "",
        Date.valueOf("2016-10-30"),
        Timestamp.valueOf("2007-09-24 12:05:55.0"),
        "fail"
      ))).asJava,
      StructType(
        Array(
          StructField("download_events", LongType, nullable = true),
          StructField("download_events_descr", StringType, nullable = true),
          StructField(dateColToBeTransformed, DateType, nullable = true),
          StructField(tsColToBeTransformed, TimestampType, nullable = true),
          StructField("label", StringType, nullable = false)
        )
      )
    )
    val sourceDF = sourceDFTmp.orderBy(rand())

    val overrides = Map(
      "labelCol" -> "label", "mlFlowLoggingFlag" -> false,
      "scalingFlag" -> true, "oneHotEncodeFlag" -> false,
      "numericBoundaries" -> Map(
        "numTrees" -> Tuple2(50.0, 100.0),
        "maxBins" -> Tuple2(10.0, 20.0),
        "maxDepth" -> Tuple2(2.0, 5.0),
        "minInfoGain" -> Tuple2(0.0, 0.03),
        "subSamplingRate" -> Tuple2(0.5, 1.0)),
      "tunerParallelism" -> 10,
      "outlierFilterFlag" -> true,
      "outlierFilterPrecision" -> 0.05,
      "outlierLowerFilterNTile" -> 0.05,
      "outlierUpperFilterNTile" -> 0.95,
      //   "tunerTrainSplitMethod" -> "kSample",
      //   "tunerKFold" -> 1,
      "tunerTrainPortion" -> 0.70,
      "tunerFirstGenerationGenePool" -> 5,
      "tunerNumberOfGenerations" -> 2,
      "tunerNumberOfParentsToRetain" -> 1,
      "tunerNumberOfMutationsPerGeneration" -> 1,
      "tunerGeneticMixing" -> 0.8,
      "tunerGenerationalMutationStrategy" -> "fixed",
      "tunerEvolutionStrategy" -> "batch",
      "pipelineDebugFlag" -> true,
      "mlFlowLoggingFlag" -> false,
      "fillConfigCardinalityLimit" -> "100"
    )

    val randomForestConfig = ConfigurationGenerator.generateConfigFromMap("RandomForest", "classifier", overrides)
    val runner = FamilyRunner(sourceDF, Array(randomForestConfig)).generateFeatureEngineeredPipeline(verbose = true)
    val outputDf = runner("RandomForest").transform(sourceDF)
    val noOfCols = outputDf.columns
    assert(noOfCols.length == 17,
      s"Feature engineered dataset's columns should have been 17, but $noOfCols were found")
    outputDf.show(100)

  }

}
