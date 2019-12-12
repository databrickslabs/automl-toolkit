package com.databricks.labs.automl

import java.util.UUID

import com.databricks.labs.automl.inference.InferencePayload
import com.databricks.labs.automl.params.ConfusionOutput
import com.databricks.labs.automl.pipeline.{
  DropColumnsTransformer,
  ZipRegisterTempTransformer
}
import com.databricks.labs.automl.utils.SchemaUtils
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.runner.RunWith
import org.scalatest._

import scala.collection.mutable.ArrayBuffer

@RunWith(classOf[org.scalatestplus.junit.JUnitRunner])
abstract class AbstractUnitSpec
    extends FlatSpec
    with Matchers
    with OptionValues
    with Inside
    with Inspectors

object AutomationUnitTestsUtil {

  lazy val sparkSession: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("providentiaml-unit-tests")
    .getOrCreate()

  sparkSession.sparkContext.setLogLevel("ERROR")

  def convertCsvToDf(csvPath: String): DataFrame = {
    sparkSession.read
      .format("csv")
      .option("header", true)
      .option("inferSchema", true)
      .load(getClass.getResource(csvPath).getPath)
  }

  def getAdultDf(): DataFrame = {
    import sparkSession.implicits._
    val adultDf = convertCsvToDf("/adult_data.csv")

    var adultDfCleaned = adultDf
    for (colName <- adultDf.columns) {
      adultDfCleaned = adultDfCleaned
        .withColumn(
          colName.split("\\s+").mkString + "_trimmed",
          trim(col(colName))
        )
        .drop(colName)
    }
    adultDfCleaned
      .withColumn("label", when($"class_trimmed" === "<=50K", 0).otherwise(1))
      .drop("class_trimmed")
  }

  def assertConfusionOutput(confusionOutput: ConfusionOutput): Unit = {
    assert(
      confusionOutput != null,
      "should have not returned null confusion output"
    )
    assert(
      confusionOutput.confusionData != null,
      "should not have returned null confusion output data"
    )
    assert(
      confusionOutput.predictionData != null,
      "should not have returned null prediction data"
    )
    assert(
      confusionOutput.confusionData.count() > 0,
      "should have more than 0 rows for confusion data"
    )
    assert(
      confusionOutput.predictionData.count() > 0,
      "should have more than 0 rows for prediction data"
    )
  }

  def assertPredOutput(inputCount: Long, predictCount: Long): Unit = {
    assert(inputCount == predictCount, "count should have matched")
  }

  def getSerializablesToTmpLocation(): String = {
    System.getProperty("java.io.tmpdir") + "/" + UUID
      .randomUUID()
      .toString + "/automl"
  }

  def getRandomForestConfig(inputDataset: DataFrame,
                            evolutionStrategy: String): AutomationRunner = {
    val rfBoundaries = Map(
      "numTrees" -> Tuple2(50.0, 1000.0),
      "maxBins" -> Tuple2(10.0, 100.0),
      "maxDepth" -> Tuple2(2.0, 20.0),
      "minInfoGain" -> Tuple2(0.0, 0.075),
      "subSamplingRate" -> Tuple2(0.5, 1.0)
    )
    new AutomationRunner(inputDataset)
      .setModelingFamily("RandomForest")
      .setLabelCol("label")
      .setFeaturesCol("features")
      .naFillOn()
      .varianceFilterOn()
      .outlierFilterOff()
      .pearsonFilterOff()
      .covarianceFilterOff()
      .oneHotEncodingOn()
      .scalingOff()
      .setStandardScalerMeanFlagOff()
      .setStandardScalerStdDevFlagOff()
      .mlFlowLoggingOff()
      .mlFlowLogArtifactsOff()
      .autoStoppingOff()
      .setFilterPrecision(0.9)
      .setParallelism(20)
      .setKFold(1)
      .setTrainPortion(0.70)
      .setTrainSplitMethod("stratified")
      .setFirstGenerationGenePool(5)
      .setNumberOfGenerations(2)
      .setNumberOfParentsToRetain(2)
      .setNumberOfMutationsPerGeneration(2)
      .setGeneticMixing(0.8)
      .setGenerationalMutationStrategy("fixed")
      .setScoringMetric("f1")
      .setFeatureImportanceCutoffType("count")
      .setFeatureImportanceCutoffValue(12.0)
      .setEvolutionStrategy(evolutionStrategy)
      .setInferenceConfigSaveLocation(
        AutomationUnitTestsUtil.getSerializablesToTmpLocation()
      )
      .setNumericBoundaries(rfBoundaries)
  }

  def getLogisticRegressionConfig(
    inputDataset: DataFrame,
    evolutionStrategy: String
  ): AutomationRunner = {
    new AutomationRunner(inputDataset)
      .setModelingFamily("LogisticRegression")
      .setLabelCol("label")
      .setFeaturesCol("features")
      .naFillOn()
      .varianceFilterOn()
      .outlierFilterOff()
      .pearsonFilterOff()
      .covarianceFilterOff()
      .oneHotEncodingOn()
      .scalingOn()
      .setStandardScalerMeanFlagOn()
      .setStandardScalerStdDevFlagOff()
      .mlFlowLoggingOff()
      .mlFlowLogArtifactsOff()
      .autoStoppingOff()
      .setFilterPrecision(0.9)
      .setParallelism(20)
      .setKFold(2)
      .setTrainPortion(0.70)
      .setTrainSplitMethod("random")
      .setFirstGenerationGenePool(5)
      .setNumberOfGenerations(2)
      .setNumberOfParentsToRetain(2)
      .setNumberOfMutationsPerGeneration(2)
      .setGeneticMixing(0.8)
      .setGenerationalMutationStrategy("fixed")
      .setFeatureImportanceCutoffType("count")
      .setFeatureImportanceCutoffValue(12.0)
      .setEvolutionStrategy(evolutionStrategy)
      .setInferenceConfigSaveLocation(
        AutomationUnitTestsUtil.getSerializablesToTmpLocation()
      )
      .setTrainSplitMethod("kSample")
  }

  def getXgBoostConfig(inputDataset: DataFrame,
                       evolutionStrategy: String): AutomationRunner = {
    new AutomationRunner(inputDataset)
      .setModelingFamily("XGBoost")
      .setLabelCol("label")
      .setFeaturesCol("features")
      .naFillOn()
      .varianceFilterOn()
      .outlierFilterOff()
      .pearsonFilterOff()
      .covarianceFilterOn()
      .oneHotEncodingOn()
      .scalingOff()
      .setStandardScalerMeanFlagOff()
      .setStandardScalerStdDevFlagOff()
      .mlFlowLoggingOff()
      .mlFlowLogArtifactsOff()
      .autoStoppingOff()
      .setFilterPrecision(0.9)
      .setParallelism(20)
      .setKFold(4)
      .setTrainPortion(0.70)
      .setTrainSplitMethod("stratified")
      .setScoringMetric("f1")
      .setFirstGenerationGenePool(5)
      .setNumberOfGenerations(2)
      .setNumberOfParentsToRetain(2)
      .setNumberOfMutationsPerGeneration(2)
      .setGeneticMixing(0.8)
      .setGenerationalMutationStrategy("fixed")
      .setFeatureImportanceCutoffType("count")
      .setFeatureImportanceCutoffValue(10.0)
      .setEvolutionStrategy(evolutionStrategy)
      .setInferenceConfigSaveLocation(
        AutomationUnitTestsUtil.getSerializablesToTmpLocation()
      )
  }

  def getMlpcConfig(inputDataset: DataFrame,
                    evolutionStrategy: String): AutomationRunner = {
    new AutomationRunner(inputDataset)
      .setModelingFamily("MLPC")
      .setLabelCol("label")
      .setFeaturesCol("features")
      .naFillOn()
      .varianceFilterOn()
      .outlierFilterOff()
      .pearsonFilterOff()
      .covarianceFilterOn()
      .oneHotEncodingOff()
      .scalingOn()
      .setStandardScalerMeanFlagOff()
      .setStandardScalerStdDevFlagOff()
      .mlFlowLoggingOff()
      .mlFlowLogArtifactsOff()
      .autoStoppingOff()
      .setFilterPrecision(0.9)
      .setParallelism(20)
      .setKFold(5)
      .setTrainPortion(0.70)
      .setTrainSplitMethod("random")
      .setScoringMetric("f1")
      .setFirstGenerationGenePool(5)
      .setNumberOfGenerations(2)
      .setNumberOfParentsToRetain(2)
      .setNumberOfMutationsPerGeneration(2)
      .setGeneticMixing(0.8)
      .setGenerationalMutationStrategy("fixed")
      .setFeatureImportanceCutoffType("count")
      .setFeatureImportanceCutoffValue(10.0)
      .setEvolutionStrategy(evolutionStrategy)
      .setInferenceConfigSaveLocation(
        AutomationUnitTestsUtil.getSerializablesToTmpLocation()
      )
  }

  def getProjectDir(): String = {
    System.getProperty("user.dir")
  }
}

case class TestVars(df: DataFrame,
                    features: Array[String],
                    tempTableName: String,
                    labelCol: String,
                    featuresCol: String = "features")

object PipelineTestUtils {
  def getTestVars(): TestVars = {
    TestVars(
      AutomationUnitTestsUtil.getAdultDf(),
      Array("age_trimmed", "workclass_trimmed", "fnlwgt_trimmed"),
      "zipRegisterTempTransformer_1",
      "label"
    )
  }

  def addZipRegisterTmpTransformerStage(
    labelCol: String,
    featuresCol: Array[String]
  ): PipelineStage = {
    new ZipRegisterTempTransformer()
      .setTempViewOriginalDatasetName(Identifiable.randomUID("zipWithId"))
      .setLabelColumn(labelCol)
      .setFeatureColumns(featuresCol)
  }

  def buildFeaturesPipelineStages(
    df: DataFrame,
    labelCol: String,
    dropColumns: Boolean = true,
    ignoreCols: Array[String] = Array.empty
  ): Array[_ <: PipelineStage] = {

    val fields = SchemaUtils.extractTypes(
      df.select(
        df.columns.filterNot(item => ignoreCols.contains(item)).map(col): _*
      ),
      labelCol
    )
    val stringFields = fields.categoricalFields
    val vectorizableFields = fields.numericFields.toArray
    val dateFields = fields.dateFields.toArray
    val timeFields = fields.timeFields.toArray
    val booleanFields = fields.booleanFields.toArray

    val stages = new ArrayBuffer[PipelineStage]

    stringFields.foreach(columnName => {
      stages += new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(SchemaUtils.generateStringIndexedColumn(columnName))
        .setHandleInvalid("keep")
    })

    stages += new DropColumnsTransformer().setInputCols(stringFields.toArray)

    val featureAssemblerInputCols: Array[String] = stringFields
      .map(item => SchemaUtils.generateStringIndexedColumn(item))
      .toArray[String] ++ vectorizableFields

    stages += new VectorAssembler()
      .setInputCols(featureAssemblerInputCols)
      .setOutputCol("features")

    if (dropColumns) {
      stages += new DropColumnsTransformer()
        .setInputCols(featureAssemblerInputCols)
    }

    stages.toArray
  }

  def saveAndLoadPipeline(stages: Array[_ <: PipelineStage],
                          dataFrame: DataFrame,
                          pipelineName: String): PipelineModel = {
    val pipelineSavePath = AutomationUnitTestsUtil
      .getProjectDir() + "/target/pipeline-tests/" + pipelineName
    val pipelineModel = new Pipeline().setStages(stages).fit(dataFrame)
    pipelineModel.transform(dataFrame)
    pipelineModel.write.overwrite().save(pipelineSavePath)
    PipelineModel.load(pipelineSavePath)
  }

  def saveAndLoadPipelineModel(pipelineModel: PipelineModel,
                               dataFrame: DataFrame,
                               pipelineName: String): PipelineModel = {
    val pipelineSavePath = AutomationUnitTestsUtil
      .getProjectDir() + "/target/pipeline-tests/" + pipelineName
    pipelineModel.transform(dataFrame)
    pipelineModel.write.overwrite().save(pipelineSavePath)
    PipelineModel.load(pipelineSavePath)
  }

  def getVectorizedFeatures(df: DataFrame,
                            labelCol: String,
                            ignoreCols: Array[String]): Array[String] = {
    val fields = SchemaUtils.extractTypes(
      df.select(
        df.columns.filterNot(item => ignoreCols.contains(item)) map col: _*
      ),
      labelCol
    )
    val stringFields = fields.categoricalFields
    val vectorizableFields = fields.numericFields.toArray
    val dateFields = fields.dateFields.toArray
    val timeFields = fields.timeFields.toArray
    val booleanFields = fields.booleanFields.toArray

    stringFields
      .map(item => SchemaUtils.generateStringIndexedColumn(item))
      .toArray[String] ++ vectorizableFields
  }

}

object InferenceUnitTestUtil {
  def generateInferencePayload(): InferencePayload = {
    val adultDataset = AutomationUnitTestsUtil.getAdultDf()
    val adultDsColumns = adultDataset.columns;
    {
      new InferencePayload {
        override def data: DataFrame = adultDataset
        override def modelingColumns: Array[String] = Array("label")
        override def allColumns: Array[String] = adultDsColumns
      }
    }
  }
}
