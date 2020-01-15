package com.databricks.labs.automl.feature

import com.databricks.labs.automl.{
  AbstractUnitSpec,
  AutomationUnitTestsUtil,
  DiscreteTestDataGenerator
}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame

class KSampleTest extends AbstractUnitSpec {

  final private val LABEL_COL = "label"
  final private val FEATURES_COL = "features"
  final private val SYNTHETIC_COL = "synthetic_ksample"
  final private val FIELDS_TO_IGNORE_IN_VECTOR = Array("automl_internal_id")
  final private val KGROUPS = 25
  final private val KMEANS_MAX_ITER = 100
  final private val KMEANS_TOLERANCE = 1E-6
  final private val KMEANS_DISTANCE_MEASUREMENT = "euclidean"
  final private val KMEANS_SEED = 42L
  final private val KMEANS_PREDICTION_COL = "kgroups_ksample"
  final private val LSH_HASH_TABLES = 10
  final private val LSH_SEED = 42L
  final private val LSH_OUTPUT_COL = "hashes_ksample"
  final private val QUORUM_COUNT = 7
  final private val MINIMUM_VECTOR_COUNT_TO_MUTATE = 1
  final private val VECTOR_MUTATION_METHOD = "random"
  final private val MUTATION_MODE = "weighted"
  final private val MUTATION_VALUE = 0.5
  final private val CARDINALITY_THRESHOLD = 20
  final private val FEATURE_FIELDS = Array("a", "b", "c")

  def createVector(df: DataFrame): DataFrame = {

    new VectorAssembler()
      .setInputCols(FEATURE_FIELDS)
      .setOutputCol(FEATURES_COL)
      .transform(df)

  }

  def getImbalance(df: DataFrame): Map[Int, Long] = {

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    df.groupBy(LABEL_COL)
      .count()
      .map(x => Map(x.getAs[Int]("label") -> x.getAs[Long]("count")))
      .collect()
      .flatten
      .toMap
  }

  it should "Correctly KSample boost minority classes in match mode" in {

    val LABEL_BALANCE_MODE = "match"
    val NUMERIC_RATIO = 0.2
    val NUMERIC_TARGET = 500

    val data = DiscreteTestDataGenerator.generateKSampleData(300)

    val featurizedData = createVector(data)

    val EXPECTED_CLASS_1_COUNT_PRE = 188
    val EXPECTED_CLASS_2_COUNT_PRE = 75
    val EXPECTED_CLASS_3_COUNT_PRE = 37

    val EXPECTED_CLASS_1_COUNT_POST = 188
    val EXPECTED_CLASS_2_COUNT_POST = 188
    val EXPECTED_CLASS_3_COUNT_POST = 188

    val upSampled = SyntheticFeatureGenerator(
      featurizedData,
      FEATURES_COL,
      LABEL_COL,
      SYNTHETIC_COL,
      FIELDS_TO_IGNORE_IN_VECTOR,
      KGROUPS,
      KMEANS_MAX_ITER,
      KMEANS_TOLERANCE,
      KMEANS_DISTANCE_MEASUREMENT,
      KMEANS_SEED,
      KMEANS_PREDICTION_COL,
      LSH_HASH_TABLES,
      LSH_SEED,
      LSH_OUTPUT_COL,
      QUORUM_COUNT,
      MINIMUM_VECTOR_COUNT_TO_MUTATE,
      VECTOR_MUTATION_METHOD,
      MUTATION_MODE,
      MUTATION_VALUE,
      LABEL_BALANCE_MODE,
      CARDINALITY_THRESHOLD,
      NUMERIC_RATIO,
      NUMERIC_TARGET
    )

    val preRowCount = featurizedData.count()
    val postRowCount = upSampled.count()

    val imbalancePre = getImbalance(featurizedData)
    val imbalancePost = getImbalance(upSampled)

    assert(
      imbalancePre(1) == EXPECTED_CLASS_1_COUNT_PRE,
      "correct class 1 count"
    )

    assert(
      imbalancePre(2) == EXPECTED_CLASS_2_COUNT_PRE,
      "correct class 2 count"
    )

    assert(
      imbalancePre(3) == EXPECTED_CLASS_3_COUNT_PRE,
      "correct class 3 count"
    )

    assert(
      imbalancePost(1) == EXPECTED_CLASS_1_COUNT_POST,
      "no modification to main class"
    )
    assert(
      imbalancePost(2) == EXPECTED_CLASS_2_COUNT_POST,
      "matched expected synthetic class count"
    )
    assert(
      imbalancePost(3) == EXPECTED_CLASS_3_COUNT_POST,
      "matched expected synthetic class count"
    )

    assert(postRowCount > preRowCount, "created rows")

  }

  it should "Correctly KSample boost minority classes in percentage mode" in {

    val LABEL_BALANCE_MODE = "percentage"
    val NUMERIC_RATIO = 0.5
    val NUMERIC_TARGET = 150

    val data = DiscreteTestDataGenerator.generateKSampleData(300)

    val featurizedData = createVector(data)

    val EXPECTED_CLASS_1_COUNT_PRE = 188
    val EXPECTED_CLASS_2_COUNT_PRE = 75
    val EXPECTED_CLASS_3_COUNT_PRE = 37

    val EXPECTED_CLASS_1_COUNT_POST = 188
    val EXPECTED_CLASS_2_COUNT_POST = 94
    val EXPECTED_CLASS_3_COUNT_POST = 94

    val upSampled = SyntheticFeatureGenerator(
      featurizedData,
      FEATURES_COL,
      LABEL_COL,
      SYNTHETIC_COL,
      FIELDS_TO_IGNORE_IN_VECTOR,
      KGROUPS,
      KMEANS_MAX_ITER,
      KMEANS_TOLERANCE,
      KMEANS_DISTANCE_MEASUREMENT,
      KMEANS_SEED,
      KMEANS_PREDICTION_COL,
      LSH_HASH_TABLES,
      LSH_SEED,
      LSH_OUTPUT_COL,
      QUORUM_COUNT,
      MINIMUM_VECTOR_COUNT_TO_MUTATE,
      VECTOR_MUTATION_METHOD,
      MUTATION_MODE,
      MUTATION_VALUE,
      LABEL_BALANCE_MODE,
      CARDINALITY_THRESHOLD,
      NUMERIC_RATIO,
      NUMERIC_TARGET
    )

    val preRowCount = featurizedData.count()
    val postRowCount = upSampled.count()

    val imbalancePre = getImbalance(featurizedData)
    val imbalancePost = getImbalance(upSampled)

    assert(
      imbalancePre(1) == EXPECTED_CLASS_1_COUNT_PRE,
      "correct class 1 count"
    )

    assert(
      imbalancePre(2) == EXPECTED_CLASS_2_COUNT_PRE,
      "correct class 2 count"
    )

    assert(
      imbalancePre(3) == EXPECTED_CLASS_3_COUNT_PRE,
      "correct class 3 count"
    )

    assert(
      imbalancePost(1) == EXPECTED_CLASS_1_COUNT_POST,
      "no modification to main class"
    )
    assert(
      imbalancePost(2) == EXPECTED_CLASS_2_COUNT_POST,
      "matched expected synthetic class count"
    )
    assert(
      imbalancePost(3) == EXPECTED_CLASS_3_COUNT_POST,
      "matched expected synthetic class count"
    )

    assert(postRowCount > preRowCount, "created rows")

  }

  it should "Correctly KSample boost minority classes in target mode" in {

    val LABEL_BALANCE_MODE = "target"
    val NUMERIC_RATIO = 0.5
    val NUMERIC_TARGET = 500

    val data = DiscreteTestDataGenerator.generateKSampleData(300)

    val featurizedData = createVector(data)

    val EXPECTED_CLASS_1_COUNT_PRE = 188
    val EXPECTED_CLASS_2_COUNT_PRE = 75
    val EXPECTED_CLASS_3_COUNT_PRE = 37

    val EXPECTED_CLASS_1_COUNT_POST = 188
    val EXPECTED_CLASS_2_COUNT_POST = 500
    val EXPECTED_CLASS_3_COUNT_POST = 500

    val upSampled = SyntheticFeatureGenerator(
      featurizedData,
      FEATURES_COL,
      LABEL_COL,
      SYNTHETIC_COL,
      FIELDS_TO_IGNORE_IN_VECTOR,
      KGROUPS,
      KMEANS_MAX_ITER,
      KMEANS_TOLERANCE,
      KMEANS_DISTANCE_MEASUREMENT,
      KMEANS_SEED,
      KMEANS_PREDICTION_COL,
      LSH_HASH_TABLES,
      LSH_SEED,
      LSH_OUTPUT_COL,
      QUORUM_COUNT,
      MINIMUM_VECTOR_COUNT_TO_MUTATE,
      VECTOR_MUTATION_METHOD,
      MUTATION_MODE,
      MUTATION_VALUE,
      LABEL_BALANCE_MODE,
      CARDINALITY_THRESHOLD,
      NUMERIC_RATIO,
      NUMERIC_TARGET
    )

    val preRowCount = featurizedData.count()
    val postRowCount = upSampled.count()

    val imbalancePre = getImbalance(featurizedData)
    val imbalancePost = getImbalance(upSampled)

    assert(
      imbalancePre(1) == EXPECTED_CLASS_1_COUNT_PRE,
      "correct class 1 count"
    )

    assert(
      imbalancePre(2) == EXPECTED_CLASS_2_COUNT_PRE,
      "correct class 2 count"
    )

    assert(
      imbalancePre(3) == EXPECTED_CLASS_3_COUNT_PRE,
      "correct class 3 count"
    )

    assert(
      imbalancePost(1) == EXPECTED_CLASS_1_COUNT_POST,
      "no modification to main class"
    )
    assert(
      imbalancePost(2) == EXPECTED_CLASS_2_COUNT_POST,
      "matched expected synthetic class count"
    )
    assert(
      imbalancePost(3) == EXPECTED_CLASS_3_COUNT_POST,
      "matched expected synthetic class count"
    )

    assert(postRowCount > preRowCount, "created rows")

  }

}
