package com.databricks.labs.automl.feature

import com.databricks.labs.automl.pipeline.FeaturePipeline
import com.databricks.labs.automl.sanitize.DataSanitizer
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}
import org.apache.spark.sql.DataFrame

class FeatureInteractionTest extends AbstractUnitSpec {

  final private val LABEL_COL = "label"
  final private val FEATURE_COL = "features"
  final private val IGNORE_FIELDS = Array("automl_internal_id")
  final private val CONTINUOUS_DISCRETIZER_BUCKET_COUNT = 25
  final private val PARALLELISM = 8

  def cleanupData: (DataFrame, Array[String], Array[String], String) = {

    val data = DiscreteTestDataGenerator.generateFeatureInteractionData(1000)

    val (sanitized, fillConfig, modelType) = new DataSanitizer(data)
      .setLabelCol(LABEL_COL)
      .setFieldsToIgnoreInVector(IGNORE_FIELDS)
      .generateCleanData()

    val (cleanData, featureFields, totalFields) = new FeaturePipeline(sanitized)
      .setLabelCol(LABEL_COL)
      .setFeatureCol(FEATURE_COL)
      .makeFeaturePipeline(IGNORE_FIELDS)

    val nominalFields = featureFields
      .filter(x => x.takeRight(3) == "_si")
      .filterNot(x => x.contains(LABEL_COL))

    val continuousFields = featureFields
      .diff(nominalFields)
      .filterNot(_.contains(LABEL_COL))
      .filterNot(_.contains(FEATURE_COL))

    (cleanData, nominalFields, continuousFields, modelType)
  }

  private def fieldCreationAssertion(expectedFields: Array[String],
                                     generatedFieldNames: Array[String]) = {

    assert(
      generatedFieldNames.forall(expectedFields.contains),
      "did not create any unexpected columns"
    )
    assert(
      expectedFields.forall(generatedFieldNames.contains),
      "creating the correct columns and retaining appropriate fields"
    )

  }

  it should "create correct interacted columns in 'all' mode" in {

    val RETENTION_MODE = "all"
    val TARGET_INTERACTION_PERCENTAGE = 1.0
    val EXPECTED_FIELDS = Array(
      "d_si",
      "f_si",
      "a",
      "b",
      "c",
      "e",
      "i_a_b",
      "i_a_c",
      "i_a_e",
      "i_b_c",
      "i_b_e",
      "i_c_e",
      "i_d_si_a",
      "i_d_si_b",
      "i_d_si_c",
      "i_d_si_e",
      "i_d_si_f_si_si",
      "i_f_si_b",
      "i_f_si_c",
      "i_f_si_a",
      "i_f_si_e",
      "automl_internal_id",
      "features",
      "label"
    )

    val (cleanData, nominalFields, continuousFields, modelType) = cleanupData

    val interacted = FeatureInteraction.interactFeatures(
      cleanData,
      nominalFields,
      continuousFields,
      modelType,
      RETENTION_MODE,
      LABEL_COL,
      FEATURE_COL,
      CONTINUOUS_DISCRETIZER_BUCKET_COUNT,
      PARALLELISM,
      TARGET_INTERACTION_PERCENTAGE
    )

    fieldCreationAssertion(EXPECTED_FIELDS, interacted.data.schema.names)
  }

  it should "create correct interacted columns in 'strict' mode" in {

    val RETENTION_MODE = "strict"
    val TARGET_INTERACTION_PERCENTAGE = 0.001
    val EXPECTED_FIELDS = Array(
      "d_si",
      "f_si",
      "a",
      "b",
      "c",
      "e",
      "i_b_c",
      "i_a_b",
      "i_a_c",
      "i_d_si_a",
      "i_d_si_c",
      "i_f_si_c",
      "i_f_si_e",
      "i_c_e",
      "i_d_si_b",
      "i_d_si_e",
      "i_f_si_a",
      "i_f_si_b",
      "automl_internal_id",
      "features",
      "label"
    )

    val (cleanData, nominalFields, continuousFields, modelType) = cleanupData

    val interacted = FeatureInteraction.interactFeatures(
      cleanData,
      nominalFields,
      continuousFields,
      modelType,
      RETENTION_MODE,
      LABEL_COL,
      FEATURE_COL,
      CONTINUOUS_DISCRETIZER_BUCKET_COUNT,
      PARALLELISM,
      TARGET_INTERACTION_PERCENTAGE
    )

    fieldCreationAssertion(EXPECTED_FIELDS, interacted.data.schema.names)

  }

  it should "create correct interacted columns in 'optimistic' mode" in {

    val RETENTION_MODE = "optimistic"
    val TARGET_INTERACTION_PERCENTAGE = -0.1
    val EXPECTED_FIELDS = Array(
      "d_si",
      "f_si",
      "a",
      "b",
      "c",
      "e",
      "i_d_si_a",
      "i_d_si_b",
      "i_d_si_c",
      "i_d_si_e",
      "i_f_si_b",
      "i_f_si_c",
      "i_f_si_a",
      "i_f_si_e",
      "automl_internal_id",
      "features",
      "label"
    )

    val (cleanData, nominalFields, continuousFields, modelType) = cleanupData

    val interacted = FeatureInteraction.interactFeatures(
      cleanData,
      nominalFields,
      continuousFields,
      modelType,
      RETENTION_MODE,
      LABEL_COL,
      FEATURE_COL,
      CONTINUOUS_DISCRETIZER_BUCKET_COUNT,
      PARALLELISM,
      TARGET_INTERACTION_PERCENTAGE
    )

    fieldCreationAssertion(EXPECTED_FIELDS, interacted.data.schema.names)

  }

}
