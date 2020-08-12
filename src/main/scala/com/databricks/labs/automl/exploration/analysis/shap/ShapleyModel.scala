package com.databricks.labs.automl.exploration.analysis.shap

import com.databricks.labs.automl.exploration.analysis.shap.tools.{
  MutatedVectors,
  ShapOutput,
  VectorSelectionUtilities
}
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.ml.classification.{
  DecisionTreeClassificationModel,
  GBTClassificationModel,
  LogisticRegressionModel,
  RandomForestClassificationModel
}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{
  DecisionTreeRegressionModel,
  GBTRegressionModel,
  LinearRegressionModel,
  RandomForestRegressionModel
}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._

import scala.collection.immutable

class ShapleyModel[T](vectorizedData: Dataset[Row],
                      model: T,
                      featureCol: String,
                      repartitionValue: Int,
                      vectorMutations: Int,
                      randomSeed: Long = 42L)
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
    * @return HashMap of feature index to averaged score for the partition's index shap values
    * @author Ben Wilson, Databricks
    * @since 0.8.0
    */
  private def scoreVectorMutations(
    featureIndex: Int,
    vectorComparisons: Array[MutatedVectors]
  ): immutable.HashMap[Int, Double] = {

    immutable.HashMap(featureIndex -> vectorComparisons.foldLeft(0.0) {
      case (s, v) =>
        val withFeaturePred = finalModel.value.predict(v.referenceIncluded)
        val withoutFeaturePred = finalModel.value.predict(v.referenceExcluded)
        s + (withFeaturePred - withoutFeaturePred)
    } / vectorComparisons.length.toDouble)
  }

  /**
    * Manual method interface for calculating the shap values per partition
    * @note DeveloperAPI
    * @return RDD[ShapOutput] for manual calculation of shap values
    * @author Ben Wilson, Databricks
    * @since 0.8.0
    */
  def calculate(): RDD[ShapOutput] = {

    val vectorOnlyRDD =
      vectorizedData.select(featureCol).repartition(repartitionValue).rdd

    vectorOnlyRDD.mapPartitionsWithIndex {
      case (i, d) => // for each partition
        val seedValue = new scala.util.Random(randomSeed)
        val vectorData = VectorSelectionUtilities.extractFeatureCollection(
          d.toArray,
          featureCol
        )
        val partitionSize = vectorData.keys.size
        val partitionIterations =
          scala.math.min(vectorMutations, partitionSize - 1)
        val totalFeatures = vectorData(0).keys.size
        val vecIndeces = (0 until totalFeatures).toList

        val indexMapping =
          (0 to totalFeatures)
            .map(i => { // for each feature index
              val indexScores = vectorData.keys
                .map( // iterate over each row
                  x => {
                    val mutationIndeces = VectorSelectionUtilities
                      .buildSelectionSet(
                        x,
                        partitionSize,
                        partitionIterations,
                        seedValue
                      )
                    val candidates = VectorSelectionUtilities
                      .mutateVectors(
                        vectorData,
                        vectorData(x),
                        i,
                        vecIndeces,
                        partitionSize,
                        mutationIndeces.toList,
                        seedValue
                      )
                    scoreVectorMutations(i, candidates)
                  }
                )
                .toArray
              partitionFeatureIndexMerge(indexScores)
            })
            .toArray

        val outputConstruct = indexMapping.flatMap(x => {
          x.map(y => ShapOutput(i, partitionSize, y._1, y._2))
        })

        outputConstruct.iterator
    }

  }

  /**
    * Private method for executing the shap calculation and collating the results from the
    * partitioned calculations as a weighted average collection based on partition row counts.
    * @return DataFrame of featureIndex and the averaged Shap values
    * @author Ben Wilson, Databricks
    * @since 0.8.0
    */
  protected[shap] def calculateAsDF(): DataFrame = {

    import spark.implicits._

    val shapPartitionedData = calculate()

    val totalRows = vectorizedData.count()

    val shapDFData = shapPartitionedData.toDF
      .withColumn("shapWeighted", col("rows") * col("shapValue"))

    shapDFData
      .groupBy("featureIndex")
      .agg((sum(col("shapWeighted")) / totalRows).alias("shapValues"))

  }

  /**
    * Public method for calculating Shap values from a model with a provided vector assembler
    * that was used to create the feature vector for the model being tested.
    *
    * @param vectorAssembler The vector assembler instance that was used to train the model being tested
    * @return DataFrame of feature names and Shap values
    * @author Ben Wilson, Databricks
    * @since 0.8.0
    */
  def getShapValuesFromModel(vectorAssembler: VectorAssembler): DataFrame = {

    import spark.implicits._

    val shapAggregation = calculateAsDF()

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

    val shapAggregation = calculateAsDF()

    val featureNames = inputColumns.zipWithIndex
      .map(x => (x._2, x._1))
      .toDF(Seq("featureIndex", "feature"): _*)

    shapAggregation
      .join(featureNames, Seq("featureIndex"), "inner")
      .select("feature", "shapValues")

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
               randomSeed: Long): ShapleyModel[T] =
    new ShapleyModel(
      vectorizedData,
      model,
      featureCol,
      repartitionValue,
      vectorMutations,
      randomSeed
    )

}
