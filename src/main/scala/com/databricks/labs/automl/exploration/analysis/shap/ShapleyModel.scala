package com.databricks.labs.automl.exploration.analysis.shap

import com.databricks.labs.automl.exploration.analysis.shap.tools.{
  MutatedVectors,
  VarianceAccumulator,
  ShapVal,
  ShapResult,
  VectorSelectionUtilities,
  EfficiencyResult
}
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.ml.classification.{
  DecisionTreeClassificationModel,
  GBTClassificationModel,
  LogisticRegressionModel,
  RandomForestClassificationModel
}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{
  DecisionTreeRegressionModel,
  GBTRegressionModel,
  LinearRegressionModel,
  RandomForestRegressionModel
}

import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{
  StructType,
  StructField,
  DoubleType
}

import scala.collection.immutable

class ShapleyModel[T](vectorizedData: Dataset[Row],
                      model: T,
                      featureCol: String,
                      repartitionValue: Int,
                      vectorMutations: Int,
                      randomSeed: Long)
    extends Serializable
    with SparkSessionWrapper {

  private val fitModel = {
    model match {
      case x: RandomForestRegressionModel =>
        x.asInstanceOf[RandomForestRegressionModel]
      case x: RandomForestClassificationModel =>
        x.asInstanceOf[RandomForestClassificationModel]
      case x: DecisionTreeRegressionModel =>
        x.asInstanceOf[DecisionTreeRegressionModel]
      case x: DecisionTreeClassificationModel =>
        x.asInstanceOf[DecisionTreeClassificationModel]
      case x: GBTRegressionModel      => x.asInstanceOf[GBTRegressionModel]
      case x: GBTClassificationModel  => x.asInstanceOf[GBTClassificationModel]
      case x: LogisticRegressionModel => x.asInstanceOf[LogisticRegressionModel]
      case x: LinearRegressionModel   => x.asInstanceOf[LinearRegressionModel]
    }
  }

  private val finalModel = spark.sparkContext.broadcast(fitModel)

  /**
    * HashMap Aggregation Method
    * @param data partition-wise aggregation data of feature index to shapley values for each feature
    * @return
    * @author Jas Bali, Databricks
    * @since 0.8.0
    */
  private def partitionFeatureIndexMerge(
    data: Array[immutable.HashMap[Int, Double]]
  ): Map[Int, Double] = {

    data.par
      .map(_.groupBy(_._1).mapValues(_.values.sum))
      .seq
      .flatten
      .groupBy(_._1)
      .mapValues(_.map(_._2).sum)
  }

  /**
    * Method for getting the per-row feature-based shap values.
    * @param featureIndex The index position of the vector currently being evaluated
    * @param vectorComparisons Array of vector positions to pull from the partition to mutate and do comparisons against
    * @return VarianceAccumulator that gives the mean estimate and standard error of per recrod shap values
    * @author Ben Wilson, Nick Senno Databricks
    * @since 0.8.0
    */
  private def scoreVectorMutations(
    featureIndex: Int,
    vectorComparisons: Array[MutatedVectors]
  ): VarianceAccumulator = {
    val mutationAgg = vectorComparisons.foldLeft(VarianceAccumulator(0.0, 0.0, 0)) {
      case (s, v) =>
        val withFeaturePred = finalModel.value.predict(v.referenceIncluded)
        val withoutFeaturePred = finalModel.value.predict(v.referenceExcluded)
        val featureDiff = withFeaturePred - withoutFeaturePred
        VarianceAccumulator(s.sum + featureDiff, s.sumSquare + featureDiff * featureDiff, s.n + 1)
    }
    mutationAgg
  }

  /**
    * Manual method interface for calculating the record level shap values per partition
    * @return Dataset[ShapResult] for manual calculation of record level shap values
    * @author Ben Wilson, Nick Senno Databricks
    * @since 0.8.0
    */
  def calculate(): Dataset[ShapResult] = {

    import spark.implicits._

    vectorizedData.select(featureCol).repartition(repartitionValue)
      .mapPartitions{p =>
        val seedValue = new scala.util.Random(randomSeed)
        val vectorData = VectorSelectionUtilities
          .extractFeatureCollection(
            p.toArray,
            featureCol
          )
        val partitionSize = vectorData.keys.size

        val totalFeatures = vectorData(0).keys.size
        val featureIndices = (0 until totalFeatures).toList

        vectorData.keys.map{r =>
          val featureVector = vectorData(r)
          val shapArray = featureIndices.map { f =>
            val sampleIndices = VectorSelectionUtilities
              .buildSelectionSet(
                r,
                partitionSize,
                vectorMutations,
                seedValue
              )
            val mutatedVectors = VectorSelectionUtilities
              .mutateVectors(
                vectorData,
                featureVector,
                f,
                featureIndices,
                partitionSize,
                sampleIndices.toList,
                seedValue
              )
            val mutatedScore = scoreVectorMutations(
              f,
              mutatedVectors
            )
            ShapVal(mutatedScore.mean, mutatedScore.standardError)
          }.toArray
          ShapResult(
            Vectors.dense(featureVector.keys.toArray.sorted.map{featureVector}),
            shapArray.map{_.value},
            shapArray.map{_.stdErr}
          )
        }.toIterator
      }
    }

  /**
   * Manual method interface for calculating the record level shap values per partition as a DataFrame
   *
   * @note WARNING - in order to get accurate results for field attribution, the ordering of the
   * field names MUST MATCH the order in which the fields were entered into the Vector Assembler phase.
   * @param inputColumns Seq[String] of field names in the order in which they were used to create the feature vector
   * @return Dataframe of original feature values and their corresponding shap values
   * @author Nick Senno, Databricks
   * @since 0.8.0
   */
  def calculateShapDF(inputColumns: Seq[String]): DataFrame = {
    val shapRDD = calculate.rdd
    val schemaFields = inputColumns.map{StructField(_, DoubleType)} ++ inputColumns.map{c => StructField(("shap_" + c), DoubleType)}
    val shapSchema = StructType(schemaFields)

    val flatShapRDD = shapRDD.map{r =>
      Row(r.features.toArray ++ r.shapleyValues: _*)
    }

    spark.createDataFrame(flatShapRDD, shapSchema)
  }

  /**
    * Private method for executing the shap calculation and collating the results from the
    * partitioned calculations average of the absolute value of shap values.
    * @return DataFrame of featureIndex and the averaged absolute value Shap values
    * @author Ben Wilson, Nick Senno Databricks
    * @since 0.8.0
    */
  protected[shap] def calculateAggregateShap(): DataFrame = {

    val shapDFData = calculate.toDF
      .select(posexplode(col("shapleyValues")))
      .withColumnRenamed("pos","featureIndex")
      .withColumnRenamed("col","value")

    shapDFData
      .groupBy("featureIndex")
      .agg(mean(abs(col("value"))).alias("shapValues"))
  }

  /**
    * Public method for calculating Shap values from a model with a provided vector assembler
    * that was used to create the feature vector for the model being tested.
    *
    * @param vectorAssembler The vector assembler instance that was used to train the model being tested
    * @return DataFrame of feature names and Shap values
    * @author Ben Wilson, Nick Senno Databricks
    * @since 0.8.0
    */
  def getShapValuesFromModel(vectorAssembler: VectorAssembler): DataFrame = {

    import spark.implicits._

    val shapAggregation = calculateAggregateShap()

    val featureNames = vectorAssembler.getInputCols.zipWithIndex
      .map(x => (x._2, x._1))
      .toSeq
      .toDF(Seq("featureIndex", "feature"): _*)

    shapAggregation
      .join(featureNames, Seq("featureIndex"), "inner")
      .select("feature", "shapValues")

  }

  /**
    * Public method for calculating Shap values from a model with a provided list of field names
    * that were used to train the model.
    * @note WARNING - in order to get accurate results for field attribution, the ordering of the
    *       field names MUST MATCH the order in which the fields were entered into the Vector Assembler phase.
    * @param inputColumns Seq[String] of field names in the order in which they were used to create the feature vector
    * @return DataFrame of feature names and Shap values
    * @author Ben Wilson, Databricks
    * @since 0.8.0
    */
  def getShapValuesFromModel(inputColumns: Seq[String]): DataFrame = {

    import spark.implicits._

    val shapAggregation = calculateAggregateShap()

    val featureNames = inputColumns.zipWithIndex
      .map(x => (x._2, x._1))
      .toDF(Seq("featureIndex", "feature"): _*)

    shapAggregation
      .join(featureNames, Seq("featureIndex"), "inner")
      .select("feature", "shapValues")

  }

  /**
   * Public method for handling the case that a Java Array List is passed in to get the
   * feature aggregated Shap values. This handles the case where the method is invoked by py4j
   * which requires a java.util.ArrayList from a Python List
   * @param inputColumns
   * @return DataFrame of feature names and Shap values
   * @author Nick Senno Databricks
   * @since 0.8.0
   */
  def getShapValuesFromModel(inputColumns: java.util.ArrayList[String]): DataFrame = {
    import scala.collection.JavaConversions._
    getShapValuesFromModel(inputColumns.toSeq)
  }

  /**
   * Private method for checking that the Shapley Values satisfy the efficiency constraint. This means that for each
   * observaction, the sum of the Shapley values is equal to the difference between the prediction for the observation
   * and the mean prediction over all observations
   * @param featureShapData: DataFrame this should be the direct output of the calculate method
   * @param tol The relative tolerance between the expected sum of Shapley Values from the Efficiency principal and actual sum of estimated Shapley Values
   * @return The number of observations that do not satisfy Efficiency within the given tolerance
   * @author Nick Senno Databricks
   * @since 0.8.0
   */
  def countEfficiency(featureShapData: DataFrame, tol: Double = 1e-3): Long = {
    import org.apache.spark.sql.functions.{abs, col, mean}
    import spark.implicits._
    
    require(tol > 0 && tol < 1, "Relative Tolerance is required to be between zero and one")
    
    // This should be the same as the original VectoriedData but with added Shapley values
    val predictedDF = fitModel.transform(featureShapData)
    val meanPrediction = predictedDF.select(mean(col("prediction"))).collect.head.getAs[Double](0)
    
    // TODO there is probably a slick way to do a fold left here but we'll kee it verbose for now
    // Could turn this into a method that returns 1 if shapleySum is outside our tolerance and zero otherwise then
    // Do a fold left sum. However, do I want to be able to keep these relative differences to return to the user?
    val efficiencyDF: Dataset[EfficiencyResult] = predictedDF.map{r =>
      val shapleySum = r.getAs[Seq[Double]]("shapleyValues").sum
      val predictionDelta = r.getAs[Double]("prediction") - meanPrediction
      
      // To prevent the denominator blowing up, if we have predictionDelta equal to zero within our tolerance we
      // simply return the shapleySum
      val relativeDiff = if (predictionDelta.abs > tol) 1.0 - shapleySum / predictionDelta else shapleySum
      
      EfficiencyResult(shapleySum, predictionDelta, relativeDiff)
    }

    efficiencyDF.where(abs(col("relativeDiff")) > tol).count

  }
}

/**
  * Companion Object for model-based Shapley calculations
  */
object ShapleyModel {

  def apply[T](vectorizedData: Dataset[Row],
               model: T,
               featureCol: String,
               repartitionValue: Int,
               vectorMutations: Int,
               randomSeed: Long = 1621L): ShapleyModel[T] =
    new ShapleyModel(
      vectorizedData,
      model,
      featureCol,
      repartitionValue,
      vectorMutations,
      randomSeed
    )

}
