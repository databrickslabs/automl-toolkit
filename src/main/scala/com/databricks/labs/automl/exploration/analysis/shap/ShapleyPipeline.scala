package com.databricks.labs.automl.exploration.analysis.shap

import com.databricks.labs.automl.exploration.analysis.common.AnalysisUtilities
import com.databricks.labs.automl.exploration.analysis.shap.tools.ShapResult
import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{
  DecisionTreeClassificationModel,
  GBTClassificationModel,
  LogisticRegressionModel,
  RandomForestClassificationModel
}
import org.apache.spark.ml.regression.{
  DecisionTreeRegressionModel,
  GBTRegressionModel,
  LinearRegressionModel,
  RandomForestRegressionModel
}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
  * Extracting Approximate Shapley Values from a PipelineModel through distributed means
  *
  * @param vectorizedData The DataFrame / Dataset that was used to train the model
  * @param pipeline PipelineModel that contains, at a minimum, a VectorAsesmbler and a supported Model type
  * @param vectorMutations The number of rows per partition that will be used to generate synthetic testing rows
  *                        for the approximate shapley algorithm
  * @param randomSeed Long value to set the seed (globally) for the random selector for rows to generate for
  *                   the vector mutations (does not guarantee or purport to provide deterministic behavior - only reduce
  *                   the non-deterministic behavior of calculating shap values in a distributed system)
  */
class ShapleyPipeline(vectorizedData: Dataset[Row],
                      pipeline: PipelineModel,
                      repartitionValue: Int,
                      vectorMutations: Int,
                      randomSeed: Long = 42L)
    extends SparkSessionWrapper
    with Serializable {

  /**
    * Extract the correct model type from the Pipeline Object so that it can be passed into DistributedShapley Object
    */
  private val fitModel = {
    AnalysisUtilities.getModelFromPipeline(pipeline).last match {
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

  /**
    * Method for calculating the record level Shapley values per partition and returning the Dataset of the ShapValues for further
    * processing
    * @note Developer API
    * @return Dataset[ShapResult] for manual interaction with the results from this package.
    * @author Ben Wilson, Databricks
    * @since 0.8.0
    */
  def calculateShapValues: Dataset[ShapResult] = {

    ShapleyModel(
      vectorizedData,
      fitModel,
      fitModel.getFeaturesCol,
      repartitionValue,
      vectorMutations,
      randomSeed
    ).calculate()

  }

  /**
    * Protected method for executing the shap calculation and returning the values as a Dataframe
    * @return DataFrame of feature index to shap value
    * @author Ben Wilson, Databricks
    * @since 0.8.0
    */
  protected[shap] def calculateShapValuesAsDF: DataFrame = {

    ShapleyModel(
      vectorizedData,
      fitModel,
      fitModel.getFeaturesCol,
      repartitionValue,
      vectorMutations,
      randomSeed
    ).calculateAggregateShap()

  }

  /**
    * Public method for getting the shap values from a pipeline and applying the feature names to the
    * feature index values to give correct original mappings of the fields to the shap values.
    * @return DataFrame of feature names and shap values
    * @author Ben Wilson, Databricks
    * @since 0.8.0
    */
  def getShapValuesFromPipeline: DataFrame = {

    import spark.implicits._

    val shapAggregation = calculateShapValuesAsDF

    val featureNames = AnalysisUtilities
      .getPipelineVectorFields(pipeline)
      .zipWithIndex
      .map(x => (x._2, x._1))
      .toSeq
      .toDF(Seq("featureIndex", "feature"): _*)

    shapAggregation
      .join(featureNames, Seq("featureIndex"), "inner")
      .select("feature", "shapValues")

  }

}

/**
  * Companion object for Shapley Pipeline Extract class
  */
object ShapleyPipeline {

  def apply(vectorizedData: Dataset[Row],
            pipeline: PipelineModel,
            reparitionValue: Int,
            vectorMutations: Int,
            randomSeed: Long): ShapleyPipeline =
    new ShapleyPipeline(
      vectorizedData,
      pipeline,
      reparitionValue,
      vectorMutations,
      randomSeed
    )

}
