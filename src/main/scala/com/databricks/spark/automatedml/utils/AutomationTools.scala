package com.databricks.spark.automatedml.utils

import com.databricks.spark.automatedml.params.{GenerationalReport, GenericModelReturn}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

trait AutomationTools extends SparkSessionWrapper {

  def extractPayload(cc: Product): Map[String, Any] = {
    val values = cc.productIterator
    cc.getClass.getDeclaredFields.map {
      _.getName -> (values.next() match {
        case p: Product if p.productArity > 0 => extractPayload(p)
        case x => x
      })
    }.toMap
  }

  def extractGenerationData(payload: Array[GenericModelReturn]): Map[Int, (Double, Double)] = {

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

  def dataPersist(preDF: DataFrame, postDF: DataFrame, cacheLevel: StorageLevel,
                  blockUnpersist: Boolean): Unit = {

    postDF.persist(cacheLevel)
    preDF.unpersist(blockUnpersist)

  }

  def fieldRemovalCompare(preFilterFields: Array[String], postFilterFields: Array[String]): List[String] = {

    preFilterFields.toList.filterNot(postFilterFields.toList.contains(_))

  }

  def extractGenerationalScores(payload: Array[GenericModelReturn], scoringOptimizationStrategy: String,
                                modelFamily: String, modelType: String): Array[GenerationalReport] = {

    val uniqueGenerations = payload.map(x => x.generation).toList.foldLeft(Nil: List[Int]) { (curr, next) =>
      if (curr contains next) curr else next :: curr
    }.sortWith(_ < _)

    val outputPayload = new ArrayBuffer[GenerationalReport]

    val generationScoringData = extractGenerationData(payload)

    for (g <- uniqueGenerations) {

      val generationData = payload.filter(_.generation == g)

      val generationSummaryData = generationScoringData(g)

      val bestGenerationRun = scoringOptimizationStrategy match {
        case "maximize" => generationData.sortWith(_.score > _.score)(0)
        case "minimize" => generationData.sortWith(_.score < _.score)(0)
        case _ => throw new UnsupportedOperationException(
          s"Optimization Strategy $scoringOptimizationStrategy is not supported.")
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
      case "maximize" => outputPayload.toArray.sortWith(_.generationMeanScore > _.generationMeanScore)
      case "minimize" => outputPayload.toArray.sortWith(_.generationMeanScore < _.generationMeanScore)
      case _ => throw new UnsupportedOperationException(
        s"Optimization Strategy $scoringOptimizationStrategy is not supported.")
    }
  }

  def generationDataFrameReport(generationalData: Array[GenerationalReport], sortingStrategy: String): DataFrame = {

    import spark.sqlContext.implicits._

    val rawDf = spark.sparkContext.parallelize(generationalData).toDF("model_family", "model_type",
      "generation", "generation_mean_score", "generation_std_dev_score")

    sortingStrategy match {
      case "maximize" => rawDf.orderBy(col("generation_mean_score").desc)
      case "minimize" => rawDf.orderBy(col("generation_mean_score").asc)
    }

  }

}