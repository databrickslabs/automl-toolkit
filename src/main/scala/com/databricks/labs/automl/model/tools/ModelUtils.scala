package com.databricks.labs.automl.model.tools

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object ModelUtils {

  final private val STRING_INDEX_SUFFIX = "_si"
  final private val OHE_SUFFIX = "_oh"
  final private val LOGGING_COL = "automl_internal_id"
  final private val EXACT_CARDINALITY_CUTOFF = 50L

  /**
    * Private method for getting the cardinality of a string indexed column to ensure that
    * @param df Source DataFrame
    * @param field field to test exact cardinality against
    * @return Integer count of distinct entries in the column
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def getExactFieldCardinality(df: DataFrame, field: String): Int = {

    df.select(field).distinct().count().toInt

  }

  /**
    * Private method for getting an approximate cardinality for numeric field types (not previously string indexed)
    * using approx distinct here due to speed and the prevention of a massive shuffle in the instance of high
    * cardinality fields.  If the approximate value is below a certain threshold, then it will be eligible for
    * exact measurement to ensure that maxBins threshold minimum will not cause an exception to be thrown.
    * @param df DataFrame for testing cardinality of fields
    * @param field Field to test cardinality for
    * @return Long - the estimated cardinality for the field
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def getApproxFieldCardinality(df: DataFrame, field: String): Long = {

    df.select(approx_count_distinct(field, 0.1).alias("approx"))
      .first()
      .getAs[Long]("approx")

  }

  /**
    * Method for readjusting the search space for tree-based algorithms to ensure that maxBins search space does not
    * initiate a model run where maxBins value is below the cardinality value of nominal fields in the data set.
    * Having a cardinality of a field that is higher than maxBins will prevent calculation of InformationGain / gini
    * for tree split calculations, since it won't be able to adequately perform the summarization of values
    * for the entropy calculation.  Resetting the search space based on the data presented for modeling will eliminate
    * the possibility of attempting to search an invalid space.
    * @param df DataFrame prepared for modeling
    * @param fieldsToIgnore fields to ignore from cardinality checks
    * @param labelCol label field (not needed for cardinality check)
    * @param featuresCol feature field (not needed for cardinality check)
    * @return An updated NumericMapping for the model's search space (where maxBins is located for the tree based
    *         algorithms)
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def resetTreeBinsSearchSpace(
    df: DataFrame,
    numericMap: Map[String, (Double, Double)],
    fieldsToIgnore: Array[String],
    labelCol: String,
    featuresCol: String
  ): Map[String, (Double, Double)] = {

    val numericFields = df.schema.names
      .filterNot(_.endsWith(STRING_INDEX_SUFFIX))
      .filterNot(_.endsWith(OHE_SUFFIX))
      .filterNot(fieldsToIgnore.contains)
      .filterNot(x => x == labelCol)
      .filterNot(x => x == featuresCol)
      .filterNot(x => x == LOGGING_COL)

    val numericNominalCandidates = numericFields.foldLeft(Array.empty[String]) {
      case (accum, x) =>
        if (getApproxFieldCardinality(df, x) < EXACT_CARDINALITY_CUTOFF)
          accum ++ Array(x)
        else accum
    }

    val categoricalFields =
      df.schema.names
        .filter(x => x.endsWith(STRING_INDEX_SUFFIX)) ++ numericNominalCandidates

    val maxBinsFloor = categoricalFields.foldLeft(0) {

      case (a, x) =>
        math.max(a, getExactFieldCardinality(df, x))
    } + 2

    val maxBinsTuple = numericMap("maxBins")

    val upperBound =
      if (maxBinsFloor > maxBinsTuple._2 - 25) maxBinsFloor + 100
      else maxBinsTuple._2

    numericMap + ("maxBins" -> (maxBinsFloor, upperBound))

  }

  def validateGBTClassifier(df: DataFrame, labelCol: String): Unit = {

    val distinctLabelValues = df.select(labelCol).distinct().count()

    distinctLabelValues match {
      case x if x > 2L =>
        throw new IllegalArgumentException(
          "GBT Classifier currently only supports binary " +
            "classification.  For multi-class, try 'trees', 'xgboost', 'randomforest', 'logistic', or 'mlpc"
        )
      case _ => None
    }
  }

}
