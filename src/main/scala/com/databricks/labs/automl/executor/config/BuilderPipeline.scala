package com.databricks.labs.automl.executor.config

import com.databricks.labs.automl.sanitize.DataSanitizer
import org.apache.spark.sql.DataFrame

class BuilderPipeline(df: DataFrame) {
//
//  var _modelsToTest: List[String] = List[String]()
//  var _modelType: String = ""
//
//  var _labelCol: String = "label"
//  var _featuresCol: String = "features"
//
//  var _modelTypeDetectionPrecisionThreshold: Double = 0.01
//
//  var _globalParallelism: Int = 20
//  var _localParallelism: Int = 5
//
//  def setModelType(value: String): this.type = {
//
//    require(allowableModelTypes.contains(value), s"$value is not a valid model type.  " +
//      s"Please submit one of: ${allowableModelTypes.mkString(", ")}")
//    _modelType = value
//    this
//  }
//
//  def setLabelCol(value: String): this.type = {
//    validateLabelColumn(value)
//    _labelCol = value
//    this
//  }
//
//  def setFeaturesCol(value: String): this.type = {
//    _featuresCol = value
//    this
//  }
//
//  def setModelsToTest(value: List[String]): this.type = {
//
//    val checkRegressors = validateModelRestrictions(value, regressorModels)
//    val checkClassifiers = validateModelRestrictions(value, classifierModels)
//
//    require(checkClassifiers | checkRegressors, s"Invalid supplied model detected! Please ensure that modelsToTest " +
//      s"submission is contained in either: \n\t Regressors: \t\t ${regressorModels.mkString(", ")}" +
//      s"\n\t Classifiers: \t\t ${classifierModels.mkString(", ")}")
//
//    _modelsToTest = value
//    this
//  }
//
//  def setGlobalParallelism(value: Int): this.type = {
//    _globalParallelism = value
//    this
//  }
//
//
//  def getLabelCol: String = _labelCol
//  def getFeaturesCol: String = _featuresCol
//  def getModelsToTest: List[String] = _modelsToTest
//  def getGlobalParallelism: Int = _globalParallelism
//
//
//
//
//
//
//
//
//  protected[config] def assignLocalParallelism(): this.type = {
//    _localParallelism = math.ceil(_globalParallelism / _modelsToTest.length).toInt
//    this
//  }
//
//  // Run this first
//  protected[config] def assertModelsForEvaluation(): Unit= {
//
//    if(_modelsToTest.isEmpty) {
//      validateModelType()
//      _modelType match {
//        case "regressor" => setModelsToTest(regressorModels)
//        case _ => setModelsToTest(classifierModels)
//      }
//    }
//  }
//
//
//  protected[config] def validateLabelColumn(label: String): Unit = {
//
//    val dfSchema = df.schema.fieldNames
//    require(dfSchema.contains(label), s"Supplied label, $label, is not contained in schema of supplied DataFrame!")
//  }
//
//
//  protected[config] def validateModelType(): this.type = {
//
//    val detectedModelType = new DataSanitizer(df)
//      .setLabelCol(_labelCol)
//      .setFilterPrecision(_modelTypeDetectionPrecisionThreshold)
//      .decideModel()
//
//    if(_modelType != "") setModelType(detectedModelType)
//    else {
//      if (_modelType != detectedModelType) {
//        println(s"WARNING! Detected model type: $detectedModelType does not match supplied model type from " +
//          s"configuration: ${_modelType} \n \t\t Ensure that model selection type is intentional.")
//      }
//    }
//  this
//  }
//
//  protected[config] def validateModelRestrictions(definedModelList: List[String], allowedModelList: List[String]):
//  Boolean = {
//
//    val validationChecks = definedModelList.map(x => allowedModelList.contains(x))
//
//    !validationChecks.contains(false)

  }

  /**
    * TODO's:
    * 1. Make an accessor class for building different configs for each family that's desired to be calculated.
    *  - Companion object to this constructor that will simply return the state of the defaults in order to override
    *  and mutate them easier?  Or stick with the current design of defaults loading upon instantiation?
    *  - Validation checking needs to be done to ensure that each of the model families can support the type of task
    *  (classification vs. regression) and that the type of data set that is supplied can support these model types.
    *  - This builder will return a configuration object instantiation of the main builder case class.
    * 2. Generate the basic types of model families as named objects
    * 3. Build a string converter that allows for abbreviations / string attribution to define the model family type
    * 4. Call the DataPrep class, then define the model runner processes in a new Automation package
    *  - This new package should be designed to be as generic as possible, calling the respective Tuner's as needed
    *  - There should be a pool of resources to run all packaged models together asynchronously
    *  - Logging to mlflow should be accomplished in each generation completion? Think about how to do this.
    * 5. Think about serializing the models in either a docker compose script or as an MLeap artifact. (FUTURE WORK)
    */


  // TODO: super-cheese way of doing this is to turn mlflow logging off, run a bunch of automation runner runs
  // and then collect the results at the end to write in one large commit to mlflow.  Will need separate model paths
  // for writing to object store, but won't need any other config changes other than a change to 'best model logging'


//
//  protected[build] def executeRandomForestRegression() = {
//
//    val dataPayload = new DataPrep(df)
//      .setLabelCol(_labelCol)
//      .setFeaturesCol(_featuresCol)
//
//
//
//  }

  //1. Construct the DataPrep config for each model family type
  //2. Construct the Model config
  //3. Instantiate a new execute the full run (without logging) for a family
  //2. Build the configs for each model type
  //3.


