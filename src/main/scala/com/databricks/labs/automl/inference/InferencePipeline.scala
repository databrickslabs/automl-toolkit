package com.databricks.labs.automl.inference

import com.databricks.labs.automl.executor.AutomationConfig
import com.databricks.labs.automl.feature.structures.NominalIndexCollection
import com.databricks.labs.automl.pipeline.FeaturePipeline
import com.databricks.labs.automl.sanitize.Scaler
import com.databricks.labs.automl.utils.{AutomationTools, DataValidation}
import ml.dmlc.xgboost4j.scala.spark.{
  XGBoostClassificationModel,
  XGBoostRegressionModel
}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import com.databricks.labs.automl.inference.InferenceConfig._
import org.apache.spark.ml.Pipeline

class InferencePipeline(df: DataFrame)
    extends AutomationConfig
    with AutomationTools
    with DataValidation
    with InferenceTools {

  /**
    * Data Prep to:
    *   - select only the initial columns that were present at the beginning of the training run
    *   - Convert the datetime entities to correct actionable types
    *   - StringIndex categorical (text or ordinal) fields
    *   - Fill NA with the values that were used during the training run for each column
    * @return The courier object InferencePayload[<DataFrame>, <ColumnsForFeatureVector>, <AllColumns>]
    */
  private def dataPreparation(): InferencePayload = {

    // Filter out any non-used fields that may be included in future data sets that weren't part of model training
//    TODO - Have to remove this temporarily
//    val initialColumnRestriction = df.select(_inferenceConfig.inferenceDataConfig.startingColumns map col:_*)

    // Build the feature Pipeline

    val featurePipelineObject = new FeaturePipeline(df, isInferenceRun = true)
      .setLabelCol(_inferenceConfig.inferenceDataConfig.labelCol)
      .setFeatureCol(_inferenceConfig.inferenceDataConfig.featuresCol)
      .setDateTimeConversionType(
        _inferenceConfig.inferenceDataConfig.dateTimeConversionType
      )

    // Get the StringIndexed DataFrame, the fields that are set for modeling, and all fields combined.
    val (indexedData, columnsForModeling, allColumns) = featurePipelineObject
      .makeFeaturePipeline(_inferenceConfig.inferenceDataConfig.fieldsToIgnore)

    val outputData = if (_inferenceConfig.inferenceSwitchSettings.naFillFlag) {
      indexedData.na
        .fill(
          _inferenceConfig.featureEngineeringConfig.naFillConfig.categoricalColumns
        )
        .na
        .fill(
          _inferenceConfig.featureEngineeringConfig.naFillConfig.numericColumns
        )
    } else {
      indexedData
    }

    createInferencePayload(outputData, columnsForModeling, allColumns)

  }

  /**
    * Helper method for creating the Feature Vector for modeling / feature engineering tasks
    * @param payload InferencePayload object that contains:
    *                - The DataFrame
    *                - The List of Columns to be included in the Feature Vector
    *                - The Full List of Columns (including ignored columns used for post-inference joining, etc.)
    * @return a new InferencePayload object (with the DataFrame now including a feature vector)
    */
  private def createFeatureVector(
    payload: InferencePayload
  ): InferencePayload = {

    val vectorAssembler = new VectorAssembler()
      .setInputCols(payload.modelingColumns)
      .setOutputCol(_inferenceConfig.inferenceDataConfig.featuresCol)

    val vectorAppliedDataFrame = vectorAssembler.transform(payload.data)

    createInferencePayload(
      vectorAppliedDataFrame,
      payload.modelingColumns,
      payload.allColumns ++ Array(
        _inferenceConfig.inferenceDataConfig.featuresCol
      )
    )

  }

  /**
    * Helper method for applying one hot encoding to the feature vector, if used in the original modeling run
    * @param payload InferencePayload object
    * @return a new InferencePayload object (the DataFrame, with and updated feature vector, and the field listings
    *         now having any previous  StringIndexed fields converted to OneHotEncoded fields.)
    */
  private def oneHotEncodingTransform(
    payload: InferencePayload
  ): InferencePayload = {

    val featurePipeline =
      new FeaturePipeline(payload.data, isInferenceRun = true)
        .setLabelCol(_inferenceConfig.inferenceDataConfig.labelCol)
        .setFeatureCol(_inferenceConfig.inferenceDataConfig.featuresCol)
        .setDateTimeConversionType(
          _inferenceConfig.inferenceDataConfig.dateTimeConversionType
        )

    val (returnData, vectorCols, allCols) = featurePipeline.applyOneHotEncoding(
      payload.modelingColumns,
      payload.allColumns
    )

    createInferencePayload(returnData, vectorCols, allCols)

  }

  /**
    * Private helper functionn for recreating the feature interaction fields that were specified during model creation
    * @param payload Previous step payload of data, columns in feature vector, and all columns
    * @return a new InferencePayload object that has the added feature interaction fields.
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  private def createFeatureInteractions(
    payload: InferencePayload
  ): InferencePayload = {

    // Interact the columns
    val interactions =
      _inferenceConfig.featureEngineeringConfig.featureInteractionConfig.interactions

    var mutatingDataFrame = payload.data

    for (c <- interactions) {
      mutatingDataFrame =
        mutatingDataFrame.withColumn(c.outputName, col(c.left) * col(c.right))
    }

    val parsedNames = interactions.map { x =>
      (x.leftDataType, x.rightDataType) match {
        case ("nominal", "nominal") =>
          NominalIndexCollection(x.outputName, indexCheck = true)
        case _ => NominalIndexCollection(x.outputName, indexCheck = false)
      }
    }

    val nominalFields = parsedNames
      .filter(x => x.indexCheck)
      .map(x => x.name)

    val indexers = nominalFields.map { x =>
      new StringIndexer()
        .setHandleInvalid("keep")
        .setInputCol(x)
        .setOutputCol(x + "_si")
    }

    val pipeline = new Pipeline().setStages(indexers).fit(mutatingDataFrame)

    val adjustedFields = parsedNames.map { x =>
      if (x.indexCheck) x.name + "_si" else x.name
    }

    createInferencePayload(
      pipeline.transform(mutatingDataFrame),
      payload.modelingColumns ++ adjustedFields,
      payload.allColumns ++ adjustedFields
    )

  }

  /**
    * Method for performing all configured FeatureEngineering tasks as set in the InferenceMainConfig
    * @param payload InferencePayload object
    * @return new InferencePayload object with all actions applied to the Dataframe and associated field listings
    *         that were originally performed in model training.
    */
  private def executeFeatureEngineering(
    payload: InferencePayload
  ): InferencePayload = {

    // Variance Filtering
    val variancePayload =
      if (_inferenceConfig.inferenceSwitchSettings.varianceFilterFlag) {

        val fieldsToRemove =
          _inferenceConfig.featureEngineeringConfig.varianceFilterConfig.fieldsRemoved

        removeArrayOfColumns(payload, fieldsToRemove)

      } else payload

    // Outlier Filtering
    val outlierPayload =
      if (_inferenceConfig.inferenceSwitchSettings.outlierFilterFlag) {

        // apply filtering in a foreach
        var outlierData = variancePayload.data

        _inferenceConfig.featureEngineeringConfig.outlierFilteringConfig.fieldRemovalMap
          .foreach { x =>
            val field = x._1
            val direction = x._2._2
            val value = x._2._1

            outlierData = direction match {
              case "greater" => outlierData.filter(col(field) <= value)
              case "lesser"  => outlierData.filter(col(field) >= value)
            }

          }

        createInferencePayload(
          outlierData,
          variancePayload.modelingColumns,
          variancePayload.allColumns
        )

      } else variancePayload

    // Covariance Filtering
    val covariancePayload =
      if (_inferenceConfig.inferenceSwitchSettings.covarianceFilterFlag) {

        val fieldsToRemove =
          _inferenceConfig.featureEngineeringConfig.covarianceFilteringConfig.fieldsRemoved

        removeArrayOfColumns(outlierPayload, fieldsToRemove)

      } else outlierPayload

    // Pearson Filtering
    val pearsonPayload =
      if (_inferenceConfig.inferenceSwitchSettings.pearsonFilterFlag) {

        val fieldsToRemove =
          _inferenceConfig.featureEngineeringConfig.pearsonFilteringConfig.fieldsRemoved

        removeArrayOfColumns(covariancePayload, fieldsToRemove)

      } else covariancePayload

    // Build the Interacted Features
    val featureInteractionPayload =
      if (_inferenceConfig.inferenceSwitchSettings.featureInteractionFlag) {
        createFeatureInteractions(pearsonPayload)
      } else pearsonPayload

    // Build the Feature Vector
    val featureVectorPayload = createFeatureVector(featureInteractionPayload)

    // OneHotEncoding
    val oneHotEncodedPayload =
      if (_inferenceConfig.inferenceSwitchSettings.oneHotEncodeFlag) {

        oneHotEncodingTransform(featureVectorPayload)

      } else featureVectorPayload

    // Scaling
    val scaledPayload =
      if (_inferenceConfig.inferenceSwitchSettings.scalingFlag) {

        val scalerConfig =
          _inferenceConfig.featureEngineeringConfig.scalingConfig

        val scaledData = new Scaler(oneHotEncodedPayload.data)
          .setFeaturesCol(_inferenceConfig.inferenceDataConfig.featuresCol)
          .setScalerType(scalerConfig.scalerType)
          .setScalerMin(scalerConfig.scalerMin)
          .setScalerMax(scalerConfig.scalerMax)
          .setStandardScalerMeanMode(scalerConfig.standardScalerMeanFlag)
          .setStandardScalerStdDevMode(scalerConfig.standardScalerStdDevFlag)
          .setPNorm(scalerConfig.pNorm)
          .scaleFeatures()

        createInferencePayload(
          scaledData,
          oneHotEncodedPayload.modelingColumns,
          oneHotEncodedPayload.allColumns
        )

      } else oneHotEncodedPayload

    // yield the Data and the Columns for the payload

    scaledPayload

  }

  /**
    * Helper method for loading and applying a transformation on the Dataframe from FeatureEngineering tasks.
    * @param data The Dataframe from feature engineering output.
    * @return A Dataframe with a prediction and/or probability column applied.
    */
  private def loadModelAndInfer(data: DataFrame): DataFrame = {

    val modelFamily = _inferenceConfig.inferenceModelConfig.modelFamily
    val modelType = _inferenceConfig.inferenceModelConfig.modelType

    val modelLoadPath = _inferenceConfig.inferenceModelConfig.modelPathLocation

    // load the model and transform the dataframe to batch predict on the data
    modelFamily match {
      case "XGBoost" =>
        modelType match {
          case "regressor" =>
            val xgboostRegressor = XGBoostRegressionModel.load(modelLoadPath)
            xgboostRegressor.transform(data)
          case "classifier" =>
            val xgboostClassifier =
              XGBoostClassificationModel.load(modelLoadPath)
            xgboostClassifier.transform(data)
        }
      case "RandomForest" =>
        modelType match {
          case "regressor" =>
            val rfRegressor = RandomForestRegressionModel.load(modelLoadPath)
            rfRegressor.transform(data)
          case "classifier" =>
            val rfClassifier =
              RandomForestClassificationModel.load(modelLoadPath)
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
            val treesClassifier =
              DecisionTreeClassificationModel.load(modelLoadPath)
            treesClassifier.transform(data)
        }
      case "MLPC" =>
        val mlpcClassifier =
          MultilayerPerceptronClassificationModel.load(modelLoadPath)
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

  /**
    * Helper method for loading the InferenceMainConfig from a DataFrame that has been written to a storage location
    * during model training. After loading the Dataframe, the value in row 1 column 1 will be extracted, converted
    * to json, converted to an instance of InferenceMainConfig, and finally used to set the current state of this
    * class' MainInferenceConfig.
    * @param inferenceDataFrameSaveLocation The storage location path of the Dataframe.
    */
  private def getAndSetConfigFromDataFrame(
    inferenceDataFrameSaveLocation: String
  ): Unit = {

    val inferenceDataFrame = spark.read.load(inferenceDataFrameSaveLocation)

    val config = extractInferenceConfigFromDataFrame(inferenceDataFrame)

    setInferenceConfig(config)

  }

  /**
    * Main private method for executing an inference run.
    * @return A Dataframe with an applied model prediction.
    */
  private def inferencePipeline(): DataFrame = {

    // Run through the Data Preparation steps as a prelude to Feature Engineering
    val prep = dataPreparation()

    // Execute the Feature Engineering that was performed during initial model training
    val featureEngineering = executeFeatureEngineering(prep)

    // Execute the model inference and return a transformed DataFrame.
    loadModelAndInfer(featureEngineering.data)

  }

  /**
    * Public method for performing an inference run from a stored InferenceConfig Dataframe location.
    * @param inferenceConfigDFPath Path on storage of where the Dataframe was written during the training run.
    * @return A Dataframe with predictions based on a pre-trained model.
    */
  def runInferenceFromStoredDataFrame(
    inferenceConfigDFPath: String
  ): DataFrame = {

    // Load the Dataframe containing the configuration and set the InferenceMainConfig
    getAndSetConfigFromDataFrame(inferenceConfigDFPath)

    inferencePipeline()

  }

  /**
    * Public method for performing an inference run from a supplied inference config string.
    * @param jsonConfig the saved inference config from a previous run as string-encoded json
    * @return A Dataframe with prediction based on a pre-trained model.
    */
  def runInferenceFromJSONConfig(jsonConfig: String): DataFrame = {

    val config = convertJsonConfigToClass(jsonConfig)

    setInferenceConfig(config)

    inferencePipeline()

  }

  def getInferenceConfig: InferenceMainConfig = _inferenceConfig

}
