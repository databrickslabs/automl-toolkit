package com.databricks.labs.automl.utils

import com.databricks.labs.automl.inference.{
  InferenceDataConfig,
  InferenceSwitchSettings
}
import com.databricks.labs.automl.params.{
  GenerationalReport,
  GenericModelReturn,
  MLPCConfig,
  MainConfig
}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import org.json4s.{Formats, _}
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.writePretty

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

trait AutomationTools extends SparkSessionWrapper {

  def extractPayload(cc: Product): Map[String, Any] = {
    val values = cc.productIterator
    cc.getClass.getDeclaredFields.map {
      _.getName -> (values.next() match {
        case p: Product if p.productArity > 0 => extractPayload(p)
        case x                                => x
      })
    }.toMap
  }

  def extractMLPCPayload(payload: MLPCConfig): Map[String, Any] = {

    Map(
      "layers" -> payload.layers.mkString(","),
      "maxIter" -> payload.maxIter,
      "solver" -> payload.solver,
      "stepSize" -> payload.stepSize,
      "tolerance" -> payload.tolerance
    )

  }

  def extractGenerationData(
    payload: Array[GenericModelReturn]
  ): Map[Int, (Double, Double)] = {

    val scoreBuffer = new ListBuffer[(Int, Double)]
    payload.foreach { x =>
      scoreBuffer += ((x.generation, x.score))
    }
    scoreBuffer.groupBy(_._1).map {
      case (k, v) =>
        val values = v.map(_._2)
        val mean = values.sum / values.size
        val res = values.map(x => scala.math.pow(x - mean, 2))
        k -> (mean, scala.math.sqrt(res.sum / res.size))
    }
  }

  def dataPersist(preDF: DataFrame,
                  postDF: DataFrame,
                  cacheLevel: StorageLevel,
                  blockUnpersist: Boolean): (DataFrame, String) = {

    postDF.persist(cacheLevel)
    val newDFRowCount = s"Row count of data: ${postDF.count()}"
    preDF.unpersist(blockUnpersist)
    (postDF, newDFRowCount)
  }

  def fieldRemovalCompare(preFilterFields: Array[String],
                          postFilterFields: Array[String]): List[String] = {

    preFilterFields.toList.filterNot(postFilterFields.toList.contains(_))

  }

  def extractGenerationalScores(
    payload: Array[GenericModelReturn],
    scoringOptimizationStrategy: String,
    modelFamily: String,
    modelType: String
  ): Array[GenerationalReport] = {

    val uniqueGenerations = payload
      .map(x => x.generation)
      .toList
      .foldLeft(Nil: List[Int]) { (curr, next) =>
        if (curr contains next) curr else next :: curr
      }
      .sortWith(_ < _)

    val outputPayload = new ArrayBuffer[GenerationalReport]

    val generationScoringData = extractGenerationData(payload)

    for (g <- uniqueGenerations) {

      val generationData = payload.filter(_.generation == g)

      val generationSummaryData = generationScoringData(g)

      val bestGenerationRun = scoringOptimizationStrategy match {
        case "maximize" => generationData.sortWith(_.score > _.score)(0)
        case "minimize" => generationData.sortWith(_.score < _.score)(0)
        case _ =>
          throw new UnsupportedOperationException(
            s"Optimization Strategy $scoringOptimizationStrategy is not supported."
          )
      }

      val bestModel = bestGenerationRun.model
      val bestParams = bestGenerationRun.hyperParams
      val bestScores = bestGenerationRun.metrics

      outputPayload += GenerationalReport(
        modelFamily = modelFamily,
        modelType = modelType,
        generation = g,
        generationMeanScore = generationSummaryData._1,
        generationStddevScore = generationSummaryData._2
      )
    }
    scoringOptimizationStrategy match {
      case "maximize" =>
        outputPayload.toArray.sortWith(
          _.generationMeanScore > _.generationMeanScore
        )
      case "minimize" =>
        outputPayload.toArray.sortWith(
          _.generationMeanScore < _.generationMeanScore
        )
      case _ =>
        throw new UnsupportedOperationException(
          s"Optimization Strategy $scoringOptimizationStrategy is not supported."
        )
    }
  }

  def generationDataFrameReport(generationalData: Array[GenerationalReport],
                                sortingStrategy: String): DataFrame = {

    import spark.sqlContext.implicits._

    val rawDf = spark.sparkContext
      .parallelize(generationalData)
      .toDF(
        "model_family",
        "model_type",
        "generation",
        "generation_mean_score",
        "generation_std_dev_score"
      )

    sortingStrategy match {
      case "maximize" => rawDf.orderBy(col("generation_mean_score").desc)
      case "minimize" => rawDf.orderBy(col("generation_mean_score").asc)
    }

  }

  def printSchema(df: DataFrame, dataName: String): String = {

    s"Schema for $dataName is: \n  ${df.schema.fieldNames.mkString(", ")}"

  }

  def printSchema(schema: Array[String], dataName: String): String = {

    s"Schema for $dataName is: \n  ${schema.mkString(", ")}"

  }

  def trainSplitValidation(trainSplitMethod: String,
                           modelSelection: String): String = {

    modelSelection match {
      case "regressor" =>
        trainSplitMethod match {
          case "stratified" =>
            println(
              "[WARNING] Stratified Method is NOT ALLOWED on Regressors. Setting to Random."
            )
            "random"
          case _ => trainSplitMethod
        }
      case _ => trainSplitMethod

    }

  }

  /**
    * Single-pass method for recording all switch settings to the InferenceConfig Object.
    * @param config MainConfig used for starting the training AutoML run
    */
  def recordInferenceSwitchSettings(
    config: MainConfig
  ): InferenceSwitchSettings = {

    // Set the switch settings
    InferenceSwitchSettings(
      naFillFlag = config.naFillFlag,
      varianceFilterFlag = config.varianceFilterFlag,
      outlierFilterFlag = config.outlierFilterFlag,
      pearsonFilterFlag = config.pearsonFilteringFlag,
      covarianceFilterFlag = config.covarianceFilteringFlag,
      oneHotEncodeFlag = config.oneHotEncodeFlag,
      scalingFlag = config.scalingFlag,
      featureInteractionFlag = config.featureInteractionFlag
    )
  }

  /**
    * Helper method for removing any of the mutators that have occurred during pre-processing of the field types
    * @param names Array: The collection of column names from the DataFrame immediately after data pre-processing
    *              tasks of type validation and conversion.
    * @since 0.5.1
    * @author Ben Wilson, Databricks
    * @return A wrapped Array of distinct field names to use for re-producability of the model for inference runs
    *         that is cleaned of the _si or _oh suffixes as a result of feature engineering tasks.
    */
  private[utils] def cleanFieldNames(names: Array[String]): Array[String] = {

    names.map { x =>
      x.takeRight(3) match {
        case "_si" => x.dropRight(3)
        case "_oh" => x.dropRight(3)
        case _     => x
      }
    }.distinct

  }

  /**
    * Helper method for generating the Inference Config object for the data configuration steps needed to perform to
    * reproduce the modeling for subsequent inference runs.
    * @param config The full main Config that is utilized for the execution of the run.
    * @param startingFields The fields that are are returned from type casting and validation (may contain artificial
    *                       suffixes for StringIndexer (_si) and OneHotEncoder(_oh).  These will be removed before
    *                       recording.
    * @since 0.4.0
    * @return and Instance of InferenceDataConfig
    */
  def recordInferenceDataConfig(
    config: MainConfig,
    startingFields: Array[String]
  ): InferenceDataConfig = {

    // Strip out any of the trailing encoding modifications that may have been done to the starting fields.

    val cleanedStartingFields = cleanFieldNames(startingFields)

    InferenceDataConfig(
      labelCol = config.labelCol,
      featuresCol = config.featuresCol,
      startingColumns = cleanedStartingFields,
      fieldsToIgnore = config.fieldsToIgnoreInVector,
      dateTimeConversionType = config.dateTimeConversionType
    )

  }

  /**
    * Provide a human-readable report into stdout and in the logs that show the configuration for a model run
    * with the key -> value relationship shown as json
    * @param config AnyRef -> a defined case class
    * @return String in the form of pretty print syntax
    */
  def prettyPrintConfig(config: AnyRef): String = {

    implicit val formats: Formats =
      Serialization.formats(hints = FullTypeHints(List(config.getClass)))
    writePretty(config)
      .replaceAll(": \\{", "\\{")
      .replaceAll(":", "->")
  }
}
