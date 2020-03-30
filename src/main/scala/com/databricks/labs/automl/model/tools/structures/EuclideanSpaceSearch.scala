package com.databricks.labs.automl.model.tools.structures

import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{
  MaxAbsScaler,
  StringIndexer,
  VectorAssembler
}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

/**
  * Provides log search space results based on HyperParameter Vector similarity to prevent too-similar post-run
  * hyper parameters from being tested, which do not provide much information gain to the modeling run.
  * @param df DataFrame containing the hyper parameters that are generated based on the PostModelingOptimization class
  * @param numericParams Array[String] numeric Parameters available for the model type
  * @param stringParams Array[String] string Parameters available for the model type
  * @param outputCount Int Desired number of output predictions to provide.
  */
class EuclideanSpaceSearch(df: DataFrame,
                           numericParams: Array[String],
                           stringParams: Array[String],
                           outputCount: Int,
                           additionalFields: Array[String] = Array[String]())
    extends Serializable
    with SparkSessionWrapper {

  private final val SI_NAME: String = "_si"
  private final val UNSCALED_VECTOR: String = "vecParams"
  private final val SCALED_VECTOR: String = "scaledVector"
  private final val DISTANCE_COL: String = "distanceEuclid"

  @transient private lazy val fullColumns
    : Seq[String] = numericParams.toSeq ++ stringParams.toSeq ++ additionalFields.toSeq

  private def euclidean(vec: Vector): UserDefinedFunction =
    udf((feat: Vector) => Vectors.sqdist(feat, vec))

  private def generateLogNTiles: Array[Double] = {
    (0 to outputCount)
      .map(_ / outputCount.toDouble)
      .toArray
      .map(x => {
        val b = math.log(1.0 / 1E-2) / (1.0 - 1E-2)
        val a = 1.0 / math.exp(b)
        a * math.exp(b * x)
      })
  }

  private def buildVectorPipeline: Pipeline = {

    val indexers = stringParams.map(
      x => new StringIndexer().setInputCol(x).setOutputCol(x + SI_NAME)
    )

    val indexFields = stringParams.map(_ + SI_NAME) ++ numericParams

    val vectorAssembler = new VectorAssembler()
      .setInputCols(indexFields)
      .setOutputCol(UNSCALED_VECTOR)

    val scaler =
      new MaxAbsScaler()
        .setInputCol(UNSCALED_VECTOR)
        .setOutputCol(SCALED_VECTOR)

    new Pipeline().setStages(indexers :+ vectorAssembler :+ scaler)
  }

  private def executePipeline: DataFrame = {

    buildVectorPipeline.fit(df).transform(df)

  }

  def searchSpace(): DataFrame = {

    val vectoredData = executePipeline

    val topRecord =
      vectoredData.take(1).map(x => x.getAs[Vector](SCALED_VECTOR)).head

    val distanceDF = vectoredData.withColumn(
      DISTANCE_COL,
      euclidean(topRecord)(col(SCALED_VECTOR))
    )

    val nTiles = generateLogNTiles

    val nTileValues = distanceDF.stat.approxQuantile(DISTANCE_COL, nTiles, 0.0)

    nTileValues
      .map(x => {
        distanceDF
          .filter(col(DISTANCE_COL) >= x)
          .sort(col(DISTANCE_COL).asc)
          .limit(1)
      })
      .reduce(_ union _)
      .select(fullColumns.map(x => col(x)): _*)

  }

}

object EuclideanSpaceSearch {

  def apply(df: DataFrame,
            numericParams: Array[String],
            stringParams: Array[String],
            outputCount: Int,
            additionalFields: Array[String] = Array[String]()): DataFrame =
    new EuclideanSpaceSearch(
      df,
      numericParams,
      stringParams,
      outputCount,
      additionalFields
    ).searchSpace()

}
