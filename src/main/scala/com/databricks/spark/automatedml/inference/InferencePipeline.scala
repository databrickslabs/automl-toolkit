package com.databricks.spark.automatedml.inference

import com.databricks.spark.automatedml.executor.AutomationConfig
import com.databricks.spark.automatedml.pipeline.FeaturePipeline
import com.databricks.spark.automatedml.sanitize.Scaler
import com.databricks.spark.automatedml.utils.{AutomationTools, DataValidation}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, GBTRegressionModel, LinearRegressionModel, RandomForestRegressionModel}
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

  private def createFeatureVector(payload: InferencePayload): InferencePayload = {

    val vectorAssembler = new VectorAssembler()
      .setInputCols(payload.modelingColumns)
      .setOutputCol(_inferenceConfig.inferenceDataConfig.featuresCol)

    val vectorAppliedDataFrame = vectorAssembler.transform(payload.data)

    createInferencePayload(vectorAppliedDataFrame, payload.modelingColumns,
      payload.allColumns ++ Array(_inferenceConfig.inferenceDataConfig.featuresCol))

  }

  private def oneHotEncodingTransform(payload: InferencePayload): InferencePayload = {

    val featurePipeline = new FeaturePipeline(payload.data)
      .setLabelCol(_inferenceConfig.inferenceDataConfig.labelCol)
      .setFeatureCol(_inferenceConfig.inferenceDataConfig.featuresCol)
      .setDateTimeConversionType(_inferenceConfig.inferenceDataConfig.dateTimeConversionType)

    val (returnData, vectorCols, allCols) = featurePipeline.applyOneHotEncoding(payload.modelingColumns,
      payload.allColumns)

    createInferencePayload(returnData, vectorCols, allCols)

  }


  private def executeFeatureEngineering(payload: InferencePayload): InferencePayload = {

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
    val featureVectorPayload = createFeatureVector(pearsonPayload)

    // OneHotEncoding
    val oneHotEncodedPayload = if (_inferenceConfig.inferenceSwitchSettings.oneHotEncodeFlag) {

      oneHotEncodingTransform(featureVectorPayload)

    } else featureVectorPayload

    // Scaling
    val scaledPayload = if (_inferenceConfig.inferenceSwitchSettings.scalingFlag) {

      val scalerConfig = _inferenceConfig.featureEngineeringConfig.scalingConfig

      val scaledData = new Scaler(oneHotEncodedPayload.data)
        .setFeaturesCol(_inferenceConfig.inferenceDataConfig.featuresCol)
        .setScalerType(scalerConfig.scalerType)
        .setScalerMin(scalerConfig.scalerMin)
        .setScalerMax(scalerConfig.scalerMax)
        .setStandardScalerMeanMode(scalerConfig.standardScalerMeanFlag)
        .setStandardScalerStdDevMode(scalerConfig.standardScalerStdDevFlag)
        .setPNorm(scalerConfig.pNorm)
        .scaleFeatures()

      createInferencePayload(scaledData, oneHotEncodedPayload.modelingColumns, oneHotEncodedPayload.allColumns)

    } else oneHotEncodedPayload

    // yield the Data and the Columns for the payload

    scaledPayload

  }

  private def loadModelAndInfer(data: DataFrame): DataFrame = {

    val modelFamily = _inferenceConfig.inferenceModelConfig.modelFamily
    val modelType = _inferenceConfig.inferenceModelConfig.modelType

    val modelLoadPath = _inferenceConfig.inferenceModelConfig.modelPathLocation

    // load the model and transform the dataframe to batch predict on the data
    modelFamily match {
      case "RandomForest" =>
        modelType match {
          case "regressor" =>
            val rfRegressor = RandomForestRegressionModel.load(modelLoadPath)
            rfRegressor.transform(data)
          case "classifier" =>
            val rfClassifier = RandomForestClassificationModel.load(modelLoadPath)
            rfClassifier.transform(data)
        }
      case "GBT" =>
        modelType match {
          case "regressor" =>
            val gbtRegressor = GBTRegressionModel.load(modelLoadPath)
            gbtRegressor.transform(data)
          case "classifier" =>
            val gbtClassifier = GBTClassificationModel.load(modelLoadPath)
            gbtClassifier.transform(data)
        }
      case "Trees" =>
        modelType match {
          case "regressor" =>
            val treesRegressor = DecisionTreeRegressionModel.load(modelLoadPath)
            treesRegressor.transform(data)
          case "classifier" =>
            val treesClassifier = DecisionTreeClassificationModel.load(modelLoadPath)
            treesClassifier.transform(data)
        }
      case "MLPC" =>
        val mlpcClassifier = MultilayerPerceptronClassificationModel.load(modelLoadPath)
        mlpcClassifier.transform(data)
      case "LinearRegression" =>
        val linearRegressor = LinearRegressionModel.load(modelLoadPath)
        linearRegressor.transform(data)
      case "LogisticRegression" =>
        val logisticRegressor = LogisticRegressionModel.load(modelLoadPath)
        logisticRegressor.transform(data)
      case "SVM" =>
        val svmClassifier = LinearSVCModel.load(modelLoadPath)
        svmClassifier.transform(data)
    }
  }

  private def getAndSetConfig(inferenceDataFrameSaveLocation: String): Unit = {

    val inferenceDataFrame = spark.read.load(inferenceDataFrameSaveLocation)

    val config = extractInferenceConfigFromDataFrame(inferenceDataFrame)

    setInferenceConfig(config)

  }

  //TODO: add in support for loading a model from mlflow with a different method call


  // TODO: load from mlflow tags



  //TODO: tag and record the inference location for each model built in mlflow.
  // Store the actual Config, as well as the location that it was written to.

  def executeInference(inferenceDataFrameSaveLocation: String): Unit = {


    //TODO: load and set the inference configuration

    val prep = dataPreparation()

    // feature engineering

    // retrieve the model

    // cast as appropriate type

    // transform the dataset

    // save the resulting data set

  }





}
