package com.databricks.spark.automatedml.inference

import com.databricks.spark.automatedml.executor.AutomationConfig
import com.databricks.spark.automatedml.pipeline.FeaturePipeline
import com.databricks.spark.automatedml.utils.{AutomationTools, DataValidation}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


class InferencePipeline(df: DataFrame) extends AutomationConfig with AutomationTools with DataValidation with
  InferenceConfig with InferenceTools {


  /**
    *
    * Replayability
    *
    * 1. Field Casting
    *     1. Need a Map of all fields and what they should be converted to
    *     2. Call base method for doing this conversion.
    * 2. NA Fill
    *     1. Map of character cols and what they should be mapped to
    *     2. Map of numeric cols and what they should be mapped to
    * 3. Variance Filter
    *     1. Array of fields to remove
    *     2. Method for removing those fields from the input Dataframe
    * 4. Outlier Filtering
    *     1. Map of thresholds to filter against
    *         1. ColumnName -> (filterValue, direction)
    *         2. Method for removing the rows that are outside of those thresholds.
    * 5. Create Feature Vector
    * 6. Covariance Filtering
    *     1. Array of columns to remove
    *     2. Re-use column filtering from variance filter
    *     3. Re-create Feature Vector
    * 7. Pearson Filtering
    *     1. Array of columns to remove
    *     2. Reuse column filtering
    *     3. Re-create Feature Vector
    * 8. OneHotEncode
    *     1. Re-create Feature Vector
    * 9. Scaling
    *     1. Re-create Feature Vector
    * 10. Model load
    *     1. need RunID as logged by MLFlow
    *     2. Retrieve model artifact
    *     3. Load as appropriate type
    *     4. Predict on main DataFrame
    *     5. Save Results
    *     6. Exit
    */

  /**
    * vectorPipeline(df) - FeaturePipeline().makeFeaturePipeline(fieldsToIgnore)
    *   - Get Schema
    *   - extractTypes
    *   - convertDateTimeFields
    *   -
    * select restriction (drop unwanted fields)
    * fillNA
    * variance filter (drop columns)
    *
    */

//TODO: update the main config for the Automation runner to require a location for writing the Inference Config to.

  // Step 1 - get the map of fields that need to be filled and fill them.
  private def dataPreparation(): InferencePayload = {

    // Filter out any non-used fields that may be included in future data sets that weren't part of model training
    val initialColumnRestriction = df.select(_inferenceConfig.inferenceDataConfig.startingColumns map col:_*)

    // Build the feature Pipeline

    val featurePipelineObject = new FeaturePipeline(initialColumnRestriction)
      .setLabelCol(_inferenceConfig.inferenceDataConfig.labelCol)
      .setFeatureCol(_inferenceConfig.inferenceDataConfig.featuresCol)
      .setDateTimeConversionType(_inferenceConfig.inferenceDataConfig.dateTimeConversionType)

    // Get the StringIndexed DataFrame, the fields that are set for modeling, and all fields combined.
    val (indexedData, columnsForModeling, allColumns) = featurePipelineObject
      .makeFeaturePipeline(_inferenceConfig.inferenceDataConfig.fieldsToIgnore)

    val outputData = if(_inferenceConfig.inferenceSwitchSettings.naFillFlag) {
      indexedData.na.fill(_inferenceConfig.featureEngineeringConfig.naFillConfig.categoricalColumns)
        .na.fill(_inferenceConfig.featureEngineeringConfig.naFillConfig.numericColumns)
    } else {
      indexedData
    }

    createInferencePayload(outputData, columnsForModeling, allColumns)

  }




  private def filterFields(payload: InferencePayload): InferencePayload = {

    // Variance Filtering
    val variancePayload = if (_inferenceConfig.inferenceSwitchSettings.varianceFilterFlag) {

      val fieldsToRemove = _inferenceConfig.featureEngineeringConfig.varianceFilterConfig.fieldsRemoved

      removeArrayOfColumns(payload, fieldsToRemove)

    } else payload

    // Outlier Filtering
    val outlierPayload = if (_inferenceConfig.inferenceSwitchSettings.outlierFilterFlag) {

      // apply filtering in a foreach
      var outlierData = variancePayload.data

      _inferenceConfig.featureEngineeringConfig.outlierFilteringConfig.fieldRemovalMap.foreach{x =>

        val field = x._1
        val direction = x._2._2
        val value = x._2._1

        outlierData = direction match {
          case "greater" => outlierData.filter(col(field) <= value)
          case "lesser" => outlierData.filter(col(field) >= value )
        }

      }

      createInferencePayload(outlierData, variancePayload.modelingColumns, variancePayload.allColumns)

    } else variancePayload

    // Covariance Filtering
    val covariancePayload = if (_inferenceConfig.inferenceSwitchSettings.covarianceFilterFlag) {

      val fieldsToRemove = _inferenceConfig.featureEngineeringConfig.covarianceFilteringConfig.fieldsRemoved

      removeArrayOfColumns(outlierPayload, fieldsToRemove)

    } else outlierPayload

    // Pearson Filtering
    val pearsonPayload = if (_inferenceConfig.inferenceSwitchSettings.pearsonFilterFlag) {

      val fieldsToRemove = _inferenceConfig.featureEngineeringConfig.pearsonFilteringConfig.fieldsRemoved

      removeArrayOfColumns(covariancePayload, fieldsToRemove)

    } else covariancePayload

    // Build the Feature Vector

    val (vectorData, vectorFields, fullDataFields) = vectorPipeline(pearsonPayload.data)



    val oneHotEncodedPayload = if (_inferenceConfig.inferenceSwitchSettings.oneHotEncodeFlag) {



    } else {

    }



    val scaledPayload = if (_inferenceConfig.inferenceSwitchSettings.scalingFlag) {

    } else {

    }

    // yield the Data and the Columns for the payload



  }


  def executeInference(inferenceSaveLocation: String): Unit = {

    val prep = dataPreparation()

    // feature engineering

    // retrieve the model

    // cast as appropriate type

    // transform the dataset

    // save the resulting data set

  }





}
