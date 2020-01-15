package com.databricks.labs.automl.feature

import com.databricks.labs.automl.pipeline.FeaturePipeline
import com.databricks.labs.automl.sanitize.DataSanitizer
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}
import org.apache.spark.sql.DataFrame

class FeatureInteractionTest extends AbstractUnitSpec {

  final private val LABEL_COL = "label"
  final private val FEATURE_COL = "features"
  final private val IGNORE_FIELDS = Array("automl_internal_id")
  final private val CONTINUOUS_DISCRETIZER_BUCKET_COUNT = 10
  final private val PARALLELISM = 8
  final private val TARGET_INTERACTION_PERCENTAGE = 1.0

  def cleanupData: (DataFrame, Array[String], Array[String], String) = {

    val data = DiscreteTestDataGenerator.generateFeatureInteractionData(1000)

    val (sanitized, fillConfig, modelType) = new DataSanitizer(data)
      .setLabelCol(LABEL_COL)
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

  it should "work" in {

    val RETENTION_MODE = "all"
    //all strict optimistic

    val (cleanData, nominalFields, continuousFields, modelType) = cleanupData

    cleanData.show(10)
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

    interacted.data.show(10)

  }
//TODO: the automlid is missing.  fix that.

  //TODO: get the output from this to be: [original fields ordering] [ interacted fields] [label] [features] change this in
  // the method FeatureInteraction.createCandidatesAndAddToVector()
}
