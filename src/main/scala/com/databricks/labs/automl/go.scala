package com.databricks.labs.automl

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.{Partitioner, SparkContext}

object go extends App {

  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local[8]")
    .appName("Databricks Automated ML")
    .getOrCreate()

  lazy val sc: SparkContext = SparkContext.getOrCreate()
  val train_df = spark.read.parquet("/Users/danieltomes/Dev/gitProjects/providentia/testData")
  val labelColumn = "TARGET"
  val modelingType = "RandomForest"
  val runExperiment = "runRF2"
  val projectName = "Issuance_Churn_Tomes"

  val rfBoundaries = Map(
    "numTrees" -> Tuple2(50.0, 500.0),
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(1.0, 7.0),
    "minInfoGain" -> Tuple2(0.00000008, 0.2),
    "subSamplingRate" -> Tuple2(0.5, 0.8)
  )

  val tCols = train_df.columns.slice(0,40) :+ labelColumn
  val trainDFSml = train_df.select(tCols map col: _*)


  val dataPrepConfig = new AutomationRunner(trainDFSml.limit(20000))
    .setLabelCol(labelColumn)
    .setFeaturesCol("features")
    .naFillOn()
    .varianceFilterOn()
    .outlierFilterOff()
    .pearsonFilterOff()
    .covarianceFilterOff()
    .oneHotEncodingOff()
    .scalingOff()
    .setScoringMetric("areaUnderROC")
    .dataPrepCachingOff()
    .setStandardScalerMeanFlagOff()
    .setStandardScalerStdDevFlagOff()
    .setFilterPrecision(0.9)
    .setCorrelationCutoffLow(-0.9996)
    .setCorrelationCutoffHigh(0.96)
    .setParallelism(10)
    .setKFold(1)
    .setTrainPortion(0.70)
    .setTrainSplitMethod("stratifyReduce")
    .setFirstGenerationGenePool(5)
    .setNumberOfGenerations(1)
    .setNumberOfParentsToRetain(2)
    .setNumberOfMutationsPerGeneration(5)
    .setGeneticMixing(0.8)
    .setGenerationalMutationStrategy("fixed")
    .setEvolutionStrategy("batch")
    .setDataReductionFactor(0.5)

  class myPartitioner(override val numPartitions: Int) extends Partitioner {
    override def getPartition(key: Any): Int = {
      val k = key.asInstanceOf[Int]
      k % numPartitions
    }

    override def equals(other: scala.Any): Boolean = {
      other match {
        case obj: myPartitioner => obj.numPartitions == numPartitions
        case _ => false
      }
    }
  }

  sc.parallelize(Seq((1,2),(3,4))).partitionBy(new myPartitioner(100))
//  val preppedData = dataPrepConfig.prepData()
//  preppedData.data.show()

//  val sampleModel = dataPrepConfig.runWithPrediction()
//  dataPrepConfig.run().


//  println(preppedDF.modelType)
//
//  preppedDF.data.rdd.map(row => row.getAs[Array[String]](0))
//
//  val dg = DataGeneration(preppedDF.data, preppedDF.fields, preppedDF.modelType)

//  val fi = new ManualRunner(dg)
//    .setModelingFamily(modelingType)
//    .setNumericBoundaries(rfBoundaries)
//    .setLabelCol(labelColumn)
//    .setFeaturesCol("features")
//    .dataPrepCachingOff()
//    .naFillOn()
//    .varianceFilterOn()
//    .outlierFilterOff()
//    .pearsonFilterOff()
//    .covarianceFilterOff()
//    .oneHotEncodingOff()
//    .scalingOff()
//    .setStandardScalerMeanFlagOff()
//    .setStandardScalerStdDevFlagOff()
//    .mlFlowLoggingOff()
//    .mlFlowLogArtifactsOff()
//    .autoStoppingOff()
//    .setFilterPrecision(0.9)
//    .setParallelism(2)
//    .setKFold(1)
//    .setTrainPortion(0.70)
//    .setTrainSplitMethod("underSample")
//    .setFirstGenerationGenePool(20)
//    .setNumberOfGenerations(5)
//    .setNumberOfParentsToRetain(2)
//    .setNumberOfMutationsPerGeneration(10)
//    .setGeneticMixing(0.8)
//    .setGenerationalMutationStrategy("fixed")
//    .setFeatureImportanceCutoffType("count")
//    .setFeatureImportanceCutoffValue(20.0)
//    .setEvolutionStrategy("batch")
//    .setTrainSplitMethod("stratifyReduce")
//    .setDataReductionFactor(0.8)
//
//  val vrun = fi.exploreFeatureImportances()

//
//  val selector = new ChiSqSelector()
//    .setNumTopFeatures(80)
//    .setFeaturesCol("features")
//    .setLabelCol("TARGET")
//    .setOutputCol("selectedFeatures")
//
//  val result = selector.fit(preppedDF).transform(preppedDF)
//
//  println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
//  result.show()
}
