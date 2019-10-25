# AutoML-Toolkit

The AutoML-Toolkit is an automated ML solution for Apache Spark.  It provides common data cleansing and feature engineering support, automated hyper-parameter tuning through distributed genetic algorithms, and model tracking integration with MLFlow.  It currently supports Supervised Learning algorithms that are provided as part of Spark Mllib.

## General Overview

The AutoML-Toolkit is a multi-layer API that can be used in several different ways:
1. Full Automation (high-level API) by using the AutomationRunner class and utilzing the .run() public method
2. Mid-level Automation through use of individual component API's (DataPrep / AutomationRunner public methods)
3. Low-level API's for Hyper-parameter tuning

## Full Automation

At the highest level of the API, the AutomationRunner, using defaults, requires only a Spark Dataframe to be supplied to the class instantiation.  Running the main method .run() will return 4 objects within a tuple:

#### Automation Return Values

##### The Generic Run Results, consisting of type: `Array[GenericModelReturn]`

```scala
case class GenericModelReturn(
                               hyperParams: Map[String, Any],
                               model: Any,
                               score: Double,
                               metrics: Map[String, Double],
                               generation: Int
                             )
```
These elements are: 
* hyperParams -> The elements that were utilized in the hyper-parameter tuning run, based on the type of model that was trained and validated.
* model -> The model artifact of the run, stored as an `[Any]` type.  Accessing this directly from this return element will require casting to the appropriate instance type 
  (e.g. ```val myBestModel = modelReturn(0).model.asInstanceOf[RandomForestRegressionModel]```)
* score -> The specified optimization score for the experiment run (i.e. default for classification is 'f1')
* metrics -> All available scores for the detected or specified model type (regression or classification). (i.e. for classification tasks, the metrics are: "f1", "weightedPrecision", "weightedRecall", and "accuracy")
* generation -> For batch configuration mode, the generation of evolution that the training and validation took place.  For continuous configuration mode, the iteration number of the individual run.

##### The Generational Average Scores as: 

* `Map[Int, (Double, Double)]`

<p>This data corresponds to Map[Generation, (Average Optimization Score for the Generation, Stddev of the Optimization Score for the Generation)]
</p>

##### A Spark Dataframe representation of the Generational Average Scores, consisting of 2 columns: 

* `Generation[Int]` 
* `Score[Double]`

##### A Spark Dataframe representation of the Generational Run Results, consisting of 5 columns: 

* model_family[String]
* model_type[String]
* generation[Int] 
* generation_mean_score[Double] 
* generation_std_dev_score[Double]

```text
NOTE: If using MLFlow integration, all of this data, in raw format, will be recorded and stored automatically.
```

## Alternate Main Accessor Methods

### `.runWithFeatureCulling()`

This method will run a RandomForest Model initially, extract the feature importances, and, depending on the culling 
settings (discussed below in this guide), will filter fields that do not meet the required threshold.  It will then 
build the model family type specified in the class instantiation and return the same elements of `.run()`.

### `.runFeatureCullingWithPrediction()`

This method, similar to the one above, will run feature culling, but in addition, the output payload will also have an
object called 'predictionData'.  

Example:
```scala
val results = new AutomationRunner(myData).runFeatureCullingWithPrediction()

display(results.predictionData)

```

This Dataframe will have the full data set that was used as an input to the training cycle (with the feature vector), 
a probability column (for classification) and a prediction column for both regressors and classifiers.

### `.generateDecisionSplits()`
This method will run a simply DecisionTree model with the sole purpose of providing the ability to extract
the decision path that the tree algorithm made graphically.  The String return will give a human-readable text block
that will show the if/elif/else block of decisions, a DataFrame of the Importances of each feature, and a tuned model.
This model can be cast appropriately within a notebook to get the visualization of the tree.

Example:
```scala
val treeSplits = new AutomationRunner(myData).generateDecisionSplits()

val model = treeSplits.model.asInstanceOf[DecisionTreeClassificationModel]
```

### `.runWithPrediction()`

This method will add an additional step to the standard `.run()` mode.  After tuning, the model will perform a 
`.transform()` on the ***full data set*** that was passed into the modeling run (after feature engineering / data 
cleanup tasks are completed).  This will add one or two additional fields: a prediction column (what the model predicted
based on the feature vector) and probability (for classification tasks).

### `.runWithConfusionReport()`

> WARNING - select this method ONLY if you are sure you are performing a classification task!

This method will return the same as `.runWithPrediction()`, with the addition of a Dataframe that
calculates a row-wise confusion report of all of the possible permutations of label and predicted values, 
giving a count of their pairwise combinations.

From this Dataframe, an efficient means of visualizing the true and false rates of classification can be done.

### Working with the Configuration
There are two main types of configurations, an `InstanceConfig` and a `MainConfig` both of which are 
managed by a `ConfigurationGenerator`. The instance config 
is just what it sounds like, it's an instance of a configuration but is not sufficient for a runner
but rather it is used to create the MainConfig which will then be passed into the runner.

```scala
// Generate an instance config which will create an instance config with appropriate defaults
// for the modelingType and Family
val conf = ConfigurationGenerator.generateDefaultConfig(modelingType, "classifier")

// Now that we have an instance config we can override all the defaults as necessary
conf.genericConfig.scoringMetric = "areaUnderROC"
conf.switchConfig.autoStoppingFlag = true
conf.tunerConfig.tunerAutoStoppingScore = 0.9
// etc

// Now we have a config the way we want to use it so we create the MainConfig that is to be used in the model
val mainConfig = ConfigurationGenerator.generateMainConfig(conf)
```

#### Example usage of the full AutomationRunner API (Basic)

```scala
import com.databricks.labs.automl.AutomationRunner

// Read data from a pre-defined Delta Table that has the labeled data in a column named 'label'
val myData: DataFrame = spark.table("my_db.my_data")

val conf = ConfigurationGenerator.generateDefaultConfig(modelingType, "classifier")
val modelConfig = ConfigurationGenerator.generateMainConfig(conf)

val runner = new AutomationRunner(myData)
  .setMainConfig(modelConfig)
  .runFeatureCullingWithPrediction()
```
This will extract the default configuration set in AutoML-Toolkit, run through feature engineering tasks, data cleanup, categorical data conversion, vectorization, modeling, and scoring.

The usage above overrides a number of the defaults (discussed in detail below).

#### Full Configuration for the AutomationRunner() class 
```scala
case class GenericConfig(
                          var labelCol: String,
                          var featuresCol: String,
                          var dateTimeConversionType: String,
                          var fieldsToIgnoreInVector: Array[String],
                          var scoringMetric: String,
                          var scoringOptimizationStrategy: String
                        )

case class FeatureEngineeringConfig(
                                     var numericFillStat: String,
                                     var characterFillStat: String,
                                     var modelSelectionDistinctThreshold: Int,
                                     var outlierFilterBounds: String,
                                     var outlierLowerFilterNTile: Double,
                                     var outlierUpperFilterNTile: Double,
                                     var outlierFilterPrecision: Double,
                                     var outlierContinuousDataThreshold: Int,
                                     var outlierFieldsToIgnore: Array[String],
                                     var pearsonFilterStatistic: String,
                                     var pearsonFilterDirection: String,
                                     var pearsonFilterManualValue: Double,
                                     var pearsonFilterMode: String,
                                     var pearsonAutoFilterNTile: Double,
                                     var covarianceCorrelationCutoffLow: Double,
                                     var covarianceCorrelationCutoffHigh: Double,
                                     var scalingType: String,
                                     var scalingMin: Double,
                                     var scalingMax: Double,
                                     var scalingStandardMeanFlag: Boolean,
                                     var scalingStdDevFlag: Boolean,
                                     var scalingPNorm: Double,
                                     var featureImportanceCutoffType: String,
                                     var featureImportanceCutoffValue: Double,
                                     var dataReductionFactor: Double
                                   )

case class SwitchConfig(
                         var naFillFlag: Boolean,
                         var varianceFilterFlag: Boolean,
                         var outlierFilterFlag: Boolean,
                         var pearsonFilterFlag: Boolean,
                         var covarianceFilterFlag: Boolean,
                         var oneHotEncodeFlag: Boolean,
                         var scalingFlag: Boolean,
                         var dataPrepCachingFlag: Boolean,
                         var autoStoppingFlag: Boolean
                       )


case class TunerConfig(
                        var tunerAutoStoppingScore: Double,
                        var tunerParallelism: Int,
                        var tunerKFold: Int,
                        var tunerTrainPortion: Double,
                        var tunerTrainSplitMethod: String,
                        var tunerTrainSplitChronologicalColumn: String,
                        var tunerTrainSplitChronologicalRandomPercentage: Double,
                        var tunerSeed: Long,
                        var tunerFirstGenerationGenePool: Int,
                        var tunerNumberOfGenerations: Int,
                        var tunerNumberOfParentsToRetain: Int,
                        var tunerNumberOfMutationsPerGeneration: Int,
                        var tunerGeneticMixing: Double,
                        var tunerGenerationalMutationStrategy: String,
                        var tunerFixedMutationValue: Int,
                        var tunerMutationMagnitudeMode: String,
                        var tunerEvolutionStrategy: String,
                        var tunerContinuousEvolutionMaxIterations: Int,
                        var tunerContinuousEvolutionStoppingScore: Double,
                        var tunerContinuousEvolutionParallelism: Int,
                        var tunerContinuousEvolutionMutationAggressiveness: Int,
                        var tunerContinuousEvolutionGeneticMixing: Double,
                        var tunerContinuousEvolutionRollingImprovingCount: Int,
                        var tunerModelSeed: Map[String, Any],
                        var tunerHyperSpaceInference: Boolean,
                        var tunerHyperSpaceInferenceCount: Int,
                        var tunerHyperSpaceModelCount: Int,
                        var tunerHyperSpaceModelType: String,
                        var tunerInitialGenerationMode: String,
                        var tunerInitialGenerationPermutationCount: Int,
                        var tunerInitialGenerationIndexMixingMode: String,
                        var tunerInitialGenerationArraySeed: Long
                      )

case class AlgorithmConfig(
                            var stringBoundaries: Map[String, List[String]],
                            var numericBoundaries: Map[String, (Double, Double)]
                          )

case class LoggingConfig(
                          var mlFlowLoggingFlag: Boolean,
                          var mlFlowLogArtifactsFlag: Boolean,
                          var mlFlowTrackingURI: String,
                          var mlFlowExperimentName: String,
                          var mlFlowAPIToken: String,
                          var mlFlowModelSaveDirectory: String,
                          var mlFlowLoggingMode: String,
                          var mlFlowBestSuffix: String,
                          var inferenceConfigSaveLocation: String
                        )


case class InstanceConfig(
                           var modelFamily: String,
                           var predictionType: String,
                           var genericConfig: GenericConfig,
                           var switchConfig: SwitchConfig,
                           var featureEngineeringConfig: FeatureEngineeringConfig,
                           var algorithmConfig: AlgorithmConfig,
                           var tunerConfig: TunerConfig,
                           var loggingConfig: LoggingConfig
                         )
```
Access to override each of the defaults that are provided is done through getters and setters that are shown below 
```text
(note that all configs avilable to override are shown here.  
Some demonstrated values below override other setters' behaviors.
(i.e. setting scaler type to 'normalize' ignores the scalerMin and scalerMax setter values;
the below example is simply used to expose all available options and would not be an optimal end-use config))
```

#### Full Main Config Defaults
Below are all the settings that can be overridden using the instance config. For example Let's say I want to turn 
on outlier filtering, override a few defaults for outlier filtering and change the scoring metric
since I have an unbalanced class and want to use AUROC instead of the default. I also want to 
change the parallelism to 24.

```scala
// Generate an instance Config
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
val conf = ConfigurationGenerator.generateDefaultConfig("RandomForest", "classifier")

// Reference the info below to override the values
conf.genericConfig.labelCol = "TARGET"
conf.switchConfig.outlierFilterFlag = true
conf.featureEngineeringConfig.outlierFilterPrecision = 0.05
conf.featureEngineeringConfig.outlierLowerFilterNTile = 0.05
conf.featureEngineeringConfig.outlierUpperFilterNTile = 0.95

// Generate the MainConfig from the instance Config
val modelMainConfig = ConfigurationGenerator.generateMainConfig(conf)

// Now Start the Runner
val runner = new AutomationRunner(train)
    .setMainConfig(modelMainConfig)
    .run()
``` 

```scala
  private[config] def genericConfig(
    predictionType: PredictionType
  ): GenericConfig = {
    val labelCol = "label"
    val featuresCol = "features"
    val dateTimeConversionType = "split"
    val fieldsToIgnoreInVector = Array.empty[String]
    val scoringMetric = familyScoringCheck(predictionType)
    val scoringOptimizationStrategy = familyScoringDirection(predictionType)
  }

  private[config] def switchConfig(family: FamilyValidator): SwitchConfig = {
    val naFillFlag = true
    val varianceFilterFlag = true
    val outlierFilterFlag = false
    val pearsonFilterFlag = false
    val covarianceFilterFlag = false
    val oheFlag = oneHotEncodeFlag(family) // Depends on Model Family
    val scaleFlag = scalingFlag(family) // Depends on Model Family
    val dataPrepCachingFlag = true
    val autoStoppingFlag = false
  }

  private[config] def algorithmConfig(
    modelType: ModelSelector
  ): AlgorithmConfig =
    AlgorithmConfig(
      stringBoundariesAssignment(modelType),
      numericBoundariesAssignment(modelType)
    )

  private[config] def featureEngineeringConfig(): FeatureEngineeringConfig = {
    val numericFillStat = "mean"
    val characterFillStat = "max"
    val modelSelectionDistinctThreshold = 50
    val outlierFilterBounds = "both"
    val outlierLowerFilterNTile = 0.02
    val outlierUpperFilterNTile = 0.98
    val outlierFilterPrecision = 0.01
    val outlierContinuousDataThreshold = 50
    val outlierFieldsToIgnore = Array.empty[String]
    val pearsonFilterStatistic = "pearsonStat"
    val pearsonFilterDirection = "greater"
    val pearsonFilterManualValue = 0.0
    val pearsonFilterMode = "auto"
    val pearsonAutoFilterNTile = 0.75
    val covarianceCorrelationCutoffLow = -0.99
    val covarianceCorrelctionCutoffHigh = 0.99
    val scalingType = "minMax"
    val scalingMin = 0.0
    val scalingMax = 1.0
    val scalingStandardMeanFlag = false
    val scalingStdDevFlag = true
    val scalingPNorm = 2.0
    val featureImportanceCutoffType = "count"
    val featureImportanceCutoffValue = 15.0
    val dataReductionFactor = 0.5
  }

  private[config] def tunerConfig(): TunerConfig = {
    val tunerAutoStoppingScore = 0.99
    val tunerParallelism = 20
    val tunerKFold = 5
    val tunerTrainPortion = 0.8
    val tunerTrainSplitMethod = "random"
    val tunerTrainSplitChronologicalColumn = ""
    val tunerTrainSplitChronologicalRandomPercentage = 0.0
    val tunerSeed = 42L
    val tunerFirstGenerationGenePool = 20
    val tunerNumberOfGenerations = 10
    val tunerNumberOfParentsToRetain = 3
    val tunerNumberOfMutationsPerGeneration = 10
    val tunerGeneticMixing = 0.7
    val tunerGenerationMutationStrategy = "linear"
    val tunerFixedMutationValue = 1
    val tunerMutationMagnitudeMode = "fixed"
    val tunerEvolutionStrategy = "batch"
    val tunerContinuousEvolutionMaxIterations = 200
    val tunerContinuousEvolutionStoppingScore = 1.0
    val tunerContinuousEvolutionParallelism = 4
    val tunerContinuousEvolutionMutationAggressiveness = 3
    val tunerContinuousEvolutionGeneticMixing = 0.7
    val tunerContinuousEvolutionRollingImprovementCount = 20
    val tunerModelSeed = Map.empty[String, Any]
    val tunerHyperSpaceInference = true
    val tunerHyperSpaceInferenceCount = 200000
    val tunerHyperSpaceModelCount = 10
    val tunerHyperSpaceModelType = "RandomForest"
    val tunerInitialGenerationMode = "random"
    val tunerInitialGenerationPermutationCount = 10
    val tunerInitialGenerationIndexMixingMode = "linear"
    val tunerInitialGenerationArraySeed = 42L
  }

  private[config] def loggingConfig(): LoggingConfig = {
    val mlFlowLoggingFlag = true
    val mlFlowLogArtifactsFlag = false
    val mlFlowTrackingURI = "hosted"
    val mlFlowExperimentName = "default"
    val mlFlowAPIToken = "default"
    val mlFlowModelSaveDirectory = "/mlflow/experiments/"
    val mlFlowLoggingMode = "full"
    val mlFlowBestSuffix = "_best"
    val inferenceSaveLocation = "/inference/"
  }
```

> For MLFlow config, ensure to check the model save directory carefully. The example above will write to the dbfs root location.
>> It is advised to ***specify a blob storage path directly***.

If at any time the current configuration is needed to be exposed, each setter, as well as the overall main Config and 
the individual module configs can be acquired by using the appropriate getter.  For example:
```scala
import com.databricks.labs.automl.AutomationRunner

val myData: DataFrame = spark.table("my_db.my_sample_data")

val fullConfig = new AutomationRunner(myData)

fullConfig.getAutoStoppingScore

```

> NOTE: MlFlow configs are not exposed via getters.  This is to prevent accessing the API Key and printing it by an end-user.

# Module Functionality

This API is split into two main portions:
* Feature Engineering / Data Prep / Vectorization
* Automated HyperParameter Tuning

Both modules use a few common variables among them that require consistency for the chained sequence of events to function correctly.
Aside from those, all others are specific to those main modules (discussed in detail below).

### Common Module Settings
###### Label Column Name

Setter: `.setLabelCol(<String>)`

```text
Default: "label"

This is the 'predicted value' for use in supervised learning.  
    If this field does not exist within the dataframe supplied, an assertion exception will be thrown 
    once a method (other than setters/getters) is called on the AutomationRunner() object.
```
###### Feature Column Name

Setter: `.setFeaturesCol(<String>)`

```text
Default: "features"

Purely cosmetic setting that ensures consistency throughout all of the modules within AutoML-Toolkit.  
    
[Future Feature] In a future planned release, new accessor methods will make this setting more relevant, 
    as a validated prediction data set will be returned along with run statistics.
```
###### Fields to ignore in vector

Setter: `.setFieldsToIgnoreInVector(<Array[String]>)`

```text
Default: Array.empty[String]

Provides a means for ignoring specific fields in the Dataframe from being included in any DataPrep feature 
engineering tasks, filtering, and exlusion from the feature vector for model tuning.

This is particularly useful if there is a need to perform follow-on joins to other data sets after model 
training and prediction is complete.
```
###### Model Family

Setter: `.setModelingFamily(<String>)`

```text
Default: "RandomForest"

Sets the modeling family (Spark Mllib) to be used to train / validate.
```

> For model families that support both Regression and Classification, the parameter value 
`.setModelDistinctThreshold(<Int>)` is used to determine which to use.  Distinct values in the label column, 
if below the `modelDistinctThreshold`, will use a Classifier flavor of the Model Family.  Otherwise, it will use the Regression Type.

Currently supported models:
* "RandomForest" - [Random Forest Classifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier) or [Random Forest Regressor](http://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-regression)
* "GBT" - [Gradient Boosted Trees Classifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier) or [Gradient Boosted Trees Regressor](http://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-regression)
* "Trees" - [Decision Tree Classifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier) or [Decision Tree Regressor](http://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-regression)
* "LinearRegression" - [Linear Regressor](http://spark.apache.org/docs/latest/ml-classification-regression.html#linear-regression)
* "LogisticRegression" - [Logistic Regressor](http://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression) (supports both Binomial and Multinomial)
* "MLPC" - [Multi-Layer Perceptron Classifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier)
* "SVM" - [Linear Support Vector Machines](http://spark.apache.org/docs/latest/ml-classification-regression.html#linear-support-vector-machine)

###### Date Time Conversion Type
```text
Default: "split"

Available options: "split", "unix"
```
This setting determines how to handle DateTime type fields.  
* In the "unix" setting mode, the datetime type is converted to `[Double]` type of the `[Long]` Unix timestamp in seconds.

* In the "split" setting mode, the date is transformed into its constituent parts and adding each part to seperate fields.
> By default, extraction is at maximum precision -> (year, month, day, hour, minute, second)


## Data Prep
The Data Prep Module is intended to automate many of the typical feature engineering tasks involved in prototyping of Machine Learning models.
It includes some necessary tasks (filling NA values and removing zero-information fields), and some optional data 
clean-up features that provide a trade-off between potential gains in <u>modeling accuracy</u> vs. <u>runtime complexity</u>.  

### Fill Null Values
```text
Default: ON
Turned off via setter .naFillOff()
```
> NOTE: It is **HIGHLY recommended** to leave this turned on.  If there are Null values in the feature vector, exceptions may be thrown.

This module allows for filling both numeric values and categorical & string values.

##### Options
```text
Member of configuration case class FillConfig()

setters: 
.setNumericFillStat(<String>)
.setCharacterFillStat(<String>) 
.setModelSelectionDistinctThreshold(<Int>)
```

###### Numeric Fill Stat
```text 
Default: "mean"
```
* For all numeric types (or date/time types that have been cast to numeric types)
* Allowable fill statistics: 
1. <b>"min"</b> - minimum sorted value from all distinct values of the field
2. <b>"25p"</b> - 25th percentile (Q1 / Lower IQR value) of the ascending sorted data field
3. <b>"mean"</b> - the mean (average) value of the data field
4. <b>"median"</b> - median (50th percentile / Q2) value of the ascending sorted data field
5. <b>"75p"</b> - 75th percentile (Q3 / Upper IQR value) of the ascending sorted data field
6. <b>"max"</b> - maximum sorted value from all distinct values of the field

###### Character Fill Stat
```text 
Default: "max"
```
* For all categorical, string, and byte type fields.
* Allowable fill statistics are the same as for numeric types.  However, it is <u> highly recommended to use either <b> "min" or "max"</b></u>
as this will fill in either the most frequent occuring (max) or least frequently occuring (min) term.

###### Model Selection Distinct Threshold
```text 
Default: 10
```
This setting is used for detecting numeric fields that are not of a continuous nature, but rather are ordinal or 
categorical.  While iterating through each numeric field, a check is made on distinct count on the feature field.  
Any fields that are <b>below</b> this threshold setting are handled as character fields and will be filled as per the 
setting of `.setCharacterFillStat()`

### Filter Zero Variance Features
```text
Default: ON
Turned off via setter .varianceFilterOff()

NOTE: It is HIGHLY recommended to leave this turned on.
Feature fields with zero information gain increase overall processing time and provide no real value to the model.
```
There are no options associated with module.

### Filter Outliers
```text
Default: OFF
Turned on via setter .outlierFitlerOn()
```

This module allows for detecting outliers from within each field, setting filter thresholds 
(either automatically or manually), and allowing for either tail reduction or two-sided reduction.
Including outliers in some families of machine learning models will result in dramatic overfitting to those values.  

> NOTE: It is recommended to turn this option ON if not using RandomForest, GBT, or Trees models.

##### Options
```text
Member of configuration case class OutlierConfig()

setters:
.setFilterBounds(<String>)
.setLowerFilterNTile(<Double>)
.setUpperFilterNTile(<Double>)
.setFilterPrecision(<Double>)
.setContinuousDataThreshold(<Int>)
.setFieldsToIgnore(<Array[String]>)
```

###### Filter Bounds
```text
Default: "both"
```
Filtering both 'tails' is only recommended if the nature of the data set's input features are of a normal distribution.
If there is skew in the distribution of the data, a left or right-tailed filter should be employed.
The allowable modes are:
1. "lower" - useful for left-tailed distributions (rare)
2. "both" - useful for normally distributed data (common)
3. "upper" - useful for right-tailed distributions (common)
<p></p>

For Further reading: [Distributions](https://en.wikipedia.org/wiki/List_of_probability_distributions#With_infinite_support)

###### Lower Filter NTile
```text
Default: 0.02
```
Filters out values (rows) that are below the specified quantile level based on a sort of the field's data in ascending order.
> Only applies to modes "both" and "lower"

###### Upper Filter NTile
```text
Default: 0.98
```
Filters out values (rows) that are above the specified quantile threshold based on the ascending sort of the field's data.
> Only applies to modes "both" and "upper"

###### Filter Precision
```text
Default: 0.01
```
Determines the level of precision in the calculation of the N-tile values of each field.  
Setting this number to a lower value will result in additional shuffling and computation.
The algorithm that uses the filter precision is `approx_count_distinct(columnName: String, rsd: Double)`.  
The lower that this value is set, the more accurate it is (setting it to 0.0 will be an exact count), but the more 
shuffling (computationally expensive) will be required to calculate the value.
> NOTE: **Restricted Value** range: 0.0 -> 1.0

###### Continuous Data Threshold
```text
Default: 50
```
Determines an exclusion filter of unique values that will be ignored if the unique count of the field's values is below the specified threshold.
> Example:

| Col1 	| Col2 	| Col3 	| Col4 	| Col5 	|
|:----:	|:----:	|:----:	|:----:	|:----:	|
|   1  	|  47  	|   3  	|   4  	|   1  	|
|   1  	|  54  	|   1  	|   0  	|  11  	|
|   1  	| 9999 	|   0  	|   0  	|   0  	|
|   0  	|   7  	|   3  	|   0  	|  11  	|
|   1  	|   1  	|   0  	|   0  	|   1  	|

> In this example data set, if the continuousDataThreshold value were to be set at 4, the ignored fields would be: Col1, Col3, Col4, and Col5.
> > Col2, having 5 unique entries, would be evaluated by the outlier filtering methodology and, provided that upper range filtering is being done, Row #3 (value entry 9999) would be filtered out with an UpperFilterNTile setting of 0.8 or lower. 

###### Fields To Ignore
```text
Default: Array("")
```
Optional configuration that allows certain fields to be exempt (ignored) by the outlier filtering processing.  
Any column names that are supplied to this setter will not be used for row filtering.

### Covariance Filtering
```text
Default: OFF
Turned on via setter .covarianceFilterOn()
```

Covariance Filtering is a Data Prep module that iterates through each element of the feature space 
(fields that are intended to be part of the feature vector), calculates the pearson correlation coefficient between each 
feature to every other feature, and provides for the ability to filter out highly positive or negatively correlated 
features to prevent fitting errors.

Further Reading: [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

> NOTE: This algorithm, although operating within a concurrent thread pool, can be costly to execute.  
In sequential mode (parallelism = 1), it is O(n * log(n)) and should be turned on only for an initial exploratory phase of determining predictive power.  

##### General Algorithm example

Given a data set:

| A 	| B 	| C 	| D 	| label |
|:----:	|:----:	|:----:	|:----:	|:----: |
|   1  	|  94 	|   5  	|   10 	|   1   |
|   2  	|   1  	|   4  	|   20 	|   0   | 
|   3  	|  22 	|   3  	|   30 	|   1   |
|   4  	|   5  	|   2  	|   40 	|   0   |
|   5  	|   5  	|   1  	|   50 	|   0   |

Each of the fields A:B:C:D would be compared to one another, the pearson value would be calculated, and a filter will occur.

* A->B 
* A->C 
* A->D 
* B->C 
* B->D 
* C->D

There is a perfect linear negative correlation present between A->C, a perfect postitive linear correlation between 
A->D, and a perfect linear negative correlation between C->D.
However, evaluation of C->D will not occur, as both C and D will be filtered out due to the correlation coefficient 
threshold.
The resultant data set from this module would be, after filtering: 

| A 	| B 	| label |
|:----:	|:----:	|:----: |
|   1  	|  94 	|   1   |
|   2  	|   1  	|   0   | 
|   3  	|  22 	|   1   |
|   4  	|   5  	|   0   |
|   5  	|   5  	|   0   |

##### Options
```text
Member of configuration case class CovarianceConfig()

setters:
.setCorrelationCutoffLow(<Double>)
.setCorrelationCutoffHigh(<Double>)
```
> NOTE: Max supported value for `.setCorrelationCutoffHigh` is 1.0

> NOTE: Max supported value for `.setCorrelationCutoffLow` is -1.0

> There are no settings for determining left / right / both sided filtering.  Instead, the cutoff values can be set to achieve this.
> > i.e. to only filter positively correlated values, apply the setting: `.setCorrelationCutoffLow(-1.0)` which would only filter
> > fields that are **exactly** negatively correlated (linear negative correlation)

###### Correlation Cutoff Low
```text
Default: -0.8
```
The setting at below which the right-hand comparison field will be filtered out of the data set, provided that the 
pearson correlation coefficient between left->right fields is below this threshold.

###### Correlation Cutoff High
```text
Default: 0.8
```
The upper positive correlation filter level.  Correlation Coefficients above this level setting will be removed from the data set.

### Pearson Filtering
```text
Default: OFF
Turned on via setter .pearsonFilterOn()
```
This module will perform validation of each field of the data set (excluding fields that have been added to 
`.setFieldsToIgnoreInVector(<Array[String]>])` and any fields that have been culled by any previous optional DataPrep 
feature engineering module) to the label column that has been set (`.setLabelCol()`).

The mechanism for comparison is a ChiSquareTest that utilizes one of three currently supported modes (listed below)

[Spark Doc - ChiSquaredTest](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.stat.ChiSquareTest$)

[Pearson's Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test)

##### Options
```text
Member of configuration case class PearsonConfig()

setters:
.setPearsonFilterStatistic(<String>)
.setPearsonFilterDirection(<String>)
.setPearsonFilterManualValue(<Double>)
.setPearsonFilterMode(<String>)
.setPearsonAutoFilterNTile(<Double>)
```

###### Pearson Filter Statistic
```text
Default: "pearsonStat"

allowable values: "pvalue", "pearsonStat", or "degreesFreedom"
```

Correlation Detection between a feature value and the label value is capable of 3 supported modes:
* Pearson Correlation ("pearsonStat")
> > Calculates the Pearson Correlation Coefficient in range of {-1.0, 1.0}
* Degrees Freedom ("degreesFreedom") 

    Additional Reading: 

    [Reduced Chi-squared statistic](https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic)

    [Generalized Chi-squared distribution](https://en.wikipedia.org/wiki/Generalized_chi-squared_distribution)
    
    [Degrees Of Freedom](https://en.wikipedia.org/wiki/Degrees_of_freedom_(statistics)#Of_random_vectors)
  
> > Calculates the Degrees of Freedom of the underlying linear subspace, in unbounded range {0, n} 
> > where n is the feature vector size.

*Before Overriding this value, ensure that a thorough understanding of this statistic is achieved.*

* p-value ("pvalue")

> > Calculates the p-value of independence from the Pearson chi-squared test in range of {0.0, 1.0}

###### Pearson Filter Direction
```text
Default: "greater"

allowable values: "greater" or "lesser"
```

Specifies whether to filter values out that are higher or lower than the target cutoff value.

###### Pearson Filter Manual Value
```text
Default: 0.0

(placeholder value)
```

Allows for manually filtering based on a hard-defined limit.

> Note: if using "manual" mode on this module, it is *imperative* to provide a valid value through this setter.

###### Pearson Filter Mode
```text
Default: "auto"

allowable values: "auto" or "manual"
```
Determines whether a manual filter value is used as a threshold, or whether a quantile-based approach (automated) 
based on the distribution of Chi-squared test results will be used to determine the threshold.

> The automated approach (using a specified N Tile) will adapt to more general problems and is recommended for getting 
measures of modeling feasibility (exploratory phase of modeling).  However, if utilizing the high-level API with a 
well-understood data set, it is recommended to override the mode, setting it to manual, and utilizing a known
acceptable threshold value for the test that is deemed acceptable.

###### Pearson Auto Filter N Tile
```text
Default: 0.75

allowable range: 0.0 > x > 1.0
```
([Q3 / Upper IQR value](https://en.wikipedia.org/wiki/Interquartile_range))


When in "auto" mode, this will reduce the feature vector by 75% of its total size, retaining only the 25% most 
important predictive-power features of the vector.

### OneHotEncoding 
```text
Default: OFF
Turned on via setter .oneHotEncodingOn()
```
Useful for all non-tree-based models (e.g. LinearRegression) for converting categorical features into a vector boolean 
space.

Details: [OneHotEncoderEstimator](http://spark.apache.org/docs/latest/ml-features.html#onehotencoderestimator)

Implementation here follows the general recommendation: Categorical (text) fields will be StringIndexed first, then 
OneHotEncoded.

> There are no options for this setting.  It either encodes the categorical features, or leaves them StringIndexed.


### Scaling

The Scaling Module provides for an automated way to set scaling on the feature vector prior to modeling.  
There are a number of ML algorithms that dramatically benefit from scaling of the features to prevent overfitting.  
Although tree-based algorithms are generally resilient to this issue, if using any other family of model, it is 
**highly recommended** to use some form of scaling.

The available scaling modes are:
* [minMax](http://spark.apache.org/docs/latest/ml-features.html#minmaxscaler)
   * Scales the feature vector to the specified min and max.  
   * Creates a Dense Vector.
* [standard](http://spark.apache.org/docs/latest/ml-features.html#standardscaler)
    * Scales the feature vector to the unit standard deviation of the feature (has options for centering around mean) 
    * If centering around mean, creates a Dense Vector.  Otherwise, it can maintain sparsity.
* [normalize](http://spark.apache.org/docs/latest/ml-features.html#normalizer)
    * Scales the feature vector to a p-norm normalization value.
* [maxAbs](http://spark.apache.org/docs/latest/ml-features.html#maxabsscaler)
    * Scales the feature vector to a range of {-1, 1} by dividing each value by the Max Absolute Value of the feature.
    * Retains Vector type.


##### Options
```text
member of configuration case class ScalingConfig()

setters:
.setScalerType("normalize")
.setScalerMin(0.0)                 
.setScalerMax(1.0)
.setPNorm(3)        
.setStandardScalerMeanFlagOff()   |  .setStandardScalerMeanFlagOn()  
.setStandardScalerStdDevFlagOff() |  .setStandardScalerStdDevFlagOn()
```

###### Scaler Type
```text
Default: "minMax"

allowable values: "minMax", "standard", "normalize", or "maxAbs"
```

Sets the scaling library to be employed in scaling the feature vector.

###### Scaler Min
```text
Default: 0.0

Only used in "minMax" mode
```

Used to set the scaling lower threshold for MinMax Scaler 
(normalizes all features in the vector to set the minimum post-processed value specified in this setter)

###### Scaler Max
```text
Default: 1.0

Only used in "minMax" mode
```

Used to set the scaling upper threshold for MinMax Scaler
(normalizes all features in the vector to set the maximum post-processed value specified in this setter)

###### P Norm
```text
Default: 2.0

Only used in "normalize" mode.
```

> NOTE: value must be >=1.0 for proper functionality in a finite vector space.

Sets the level of "smoothing" for scaling the noise out of the vector.  

Further Reading: [P-Norm](https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm), [L<sup>p</sup> Space](https://en.wikipedia.org/wiki/Lp_space)

###### Standard Scaler Mean Flag
```text
Default: false

Only used in "standard" mode
```

With this flag set to `true`, The features within the vector are centered around mean (0 adjusted) before scaling.
> Read the [docs](http://spark.apache.org/docs/latest/ml-features.html#standardscaler) before switching this on.
> > Setting to 'on' will create a dense vector, which will increase memory footprint of the data set.
###### Standard Scaler StdDev Flag
```text
Default: true

Only used in "standard" mode
```

Scales the data to the unit standard deviation. [Explanation](https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation)


## AutoML (Hyper-parameter concurrent genetic tuning)

The implementation for hyper parameter tuning in the AutoML-Toolkit is through the use of a genetic algorithm.  
There are currently two different modes: **Batch Mode** and **Continuous Mode**.

#### Generic Configurations

###### K Fold

This setting determines the number of splits that are done with the train / test data sets, minimizing the chance of 
overfitting during training and validation.
```text
Default: 5
```
> It is highly advised to not set this lower than 3.

> [WARNING] If using "chronological" mode with `.setTrainSplitMethod()`, ensure that 
`.setTrainSplitChronologicalRandomPercentage()` is overridden from its default of 0.0.

###### Train Portion

Sets the proportion of the input DataFrame to be used for Train (the value of this variable) and Test 
(1 - the value of this variable)
```text
Default: 0.8
```

###### Train Split Method

This setting allows for specifying how to split the provided data set into test/training sets for scoring validation.
Some ML use cases are highly time-dependent, even in traditional ML algorithms (i.e. predicting customer churn).  
As such, it is important to be able to predict on apriori data and synthetically 'predict the future' by doing validation
testing on a holdout data set that is more recent than the training data used to build the model.

Additional reading: [Sampling in Machine Learning](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)

Setter: `.setTrainSplitMethod(<String>)`
```text
Default: "random"

Available options: "random" or "chronological" or "stratified" or "underSample" or "overSample"
```
Chronological Split Mode
- Splits train / test between a sortable field (date, datetime, unixtime, etc.)
> Chronological split method **does not require** a date type or datetime type field. Any sort-able / continuous distributed field will work.

Random Split Mode
- Leaving the default value of "random" will randomly shuffle the train and test data sets each k-fold iteration.

***This is only recommended for classification data sets where there is relatively balanced counts of unique classes in the label column***
> [NOTE]: attempting to run any mode other than "random" or "chronological" on a regression problem will not work.  Default behavior
will reset the trainSplitMethod to "random" if they are selected on a regression problem.

Stratified Mode

- Stratified mode will balance all of the values present in the label column of a classification algorithm so that there
is adequate coverage of all available labels in both train and test for each kfold split step.

***It is HIGHLY RECOMMENDED to use this mode if there is a large skew in your label column and there is a need for 
training on the full, unmanipulated data set.***

UnderSampling Mode

- Under sampling will evaluate the classes within the label column and sample all classes within each kfold / model run
to target the row count of the smallest class. (Only recommended for moderate skew cases)

***If using this mode, ensure that the smallest class has sufficient row counts to make training effective!***

OverSampling Mode

- Over sampling will evaluate the class with the highest count of entries and during splitting, will replicate all 
other classes' data to *generally match* the counts of the most frequent class. (**this is not exact and can vary from 
run to run**)

***WARNING*** - using this mode will dramatically increase the training and test data set sizes.  
***Small count classes' data will be copied multiple times to eliminate skew***


###### Train Split Chronological Column

Specify the field to be used in restricting the train / test split based on sort order and percentage of data set to conduct the split.
> As specified above, there is no requirement that this field be a date or datetime type.  However, it *is recommended*.

Setter: `.setTrainSplitChronologicalColumn(<String>)`

```text
Default: "datetime"

This is a placeholder value.
Validation will occur when modeling begins (post data-prep) to ensure that this field exists in the data set.
```
> It is ***imperative*** that this field exists in the raw DataFrame being supplied to the main class. ***CASE SENSITIVE MATCH***
> > Failing to ensure this setting is correctly applied could result in an exception being thrown mid-run, wasting time and resources.

###### Train Split Chronological Random Percentage

Due to the fact that a Chronological split, when done by a sort and percentage 'take' of the DataFrame, each k-fold 
generation would extract an identical train and test data set each iteration if the split were left static.  This
setting allows for a 'jitter' to the train / test boundary to ensure that k-fold validation provides more useful results.

Setter: `.setTrainSplitChronologicalRandomPercentage(<Double>)` representing the ***percentage value*** (fractional * 100)
```text
Default: 0.0

This is a placeholder value.
```
> [WARNING] Failing to override this value if using "chronological" mode on `.setTrainSplitMethod()` is equivalent to setting 
`.setKFold(1)` for efficacy purposes, and will simply waste resources by fitting multiple copies of the same hyper 
parameters on the exact same data set.

###### First Generation Gene Pool
Determines the random search seed pool for the genetic algorithm to operate from.  
There are space constraints on numeric hyper parameters, character, and boolean that are distinct for each modeling 
family and model type.
Setting this value higher increases the chances of minimizing convergence, at the expense of a longer run time.

> Setting this value below 10 is ***not recommended***.  Values less than 6 are not permitted and will throw an assertion exception.

Setter: `.setFirstGenerationGenePool(<Int>)`

```text
Default: 20
```

###### Seed

Sets the seed for both random selection generation for the initial random pool of values, as well as initializing 
another randomizer for random train/test splits.

Setter: `.setSeed(<Long>)`

```text
Default: 42L

```

###### Model Seed

There are two setters that allow for 'jump-starting' a model tuning run, primarily for the purpose of 
fine-tuning after a previous run, or for retraining a previously trained model.

> ***CAUTION*** : using a model Seed from an initial starting point on a data set (during exploratory phase) **may** 
result in poor hyper parameter tuning performance.  *It is always best to not seed a new model*.

Setters: 
1. `.setModelSeedString(<String>)` - this is designed to take a GenericModelReturn String from a previous run (either
from stdout, log4j, or MLFlow tags) and use it as the basis for the first-round "best model" seed for the algorithm.
2. `.setModelSeedMap(<Map[String, Any]>)` - this is the more 'scala-esque' way of submitting a configuration.  
> See example in the top section of this README for guidance on syntax. ***Ensure that the Map configuration matches the
model family that is being tuned***.

###### Evolution Strategy
Determining the mode (batch vs. continuous) is done through setting the parameter `.setEvolutionStrategy(<String>)`

```text
Default: "batch"

Available options: "batch" or "continuous"
```
> Sets the mutation methodology used for optimizing the hyper parameters for the model.

### Batch Mode

In batch mode, the hyper parameter space is explored with an initial seed pool (based on Random Search with constraints).

After this initial pool is evaluated (in parallel), the best n parents from this seed generation are used to 'sire' a new generation.

This continues for as many generations are specified through the config `.setNumberOfGenerations(<Int>)`, 
*or until a stopping threshold is reached at the conclusion of a concurrent generation batch run.*

#### Batch-Specific Configurations

###### Parallelism
Sets the number of concurrent models that will be evaluated in parallel through Futures.  
This creates a new [ForkJoinPool](https://java-8-tips.readthedocs.io/en/stable/forkjoin.html), and as such, it is important to not set it too high, in order to prevent overloading
the driver JVM with too many elements in the `Array[DEqueue[task]]` ForkJoinPool.  
NOTE: There is a global limit of 32767 threads on the JVM, as well.
However, in practice, running too many models in parallel will likely OOM the workers anyway.  

>Recommended is in the {4, 500} range, depending on the model and size of the test/training sets.

Setter: `.setParallelism(<Int>)`

```text
Default: 20
```

###### Number of Generations

This setting, applied only to batch processing mode, sets the number of mutation generations that will occur.

> The higher this number, the better the exploration of the hyper parameter space will occur, although it comes at the 
expense of longer run-time.  This is a *sequential blocking* setting.  Parallelism does not effect this.

Setter: `.setNumberOfGenerations(<Int>)`

```text
Default: 10
```

###### Number of Parents To Retain

This setting will restrict the number of candidate 'best' results of the previous generation of hyper parameter tuning,
using these result's configuration to mutate the next generation of attempts.

> The higher this setting, the more 'space exploration' that will occur.  However, it may slow the possibility of 
converging to an optimal condition.

Setter: `.setNumberOfParentsToRetain(<Int>)`

```text
Default: 3
```

###### Number of Mutations Per Generation

This setting specifies the size of each evolution batch pool per generation (other than the first seed generation).

> The higher this setting is set, the more alternative spaces are checked, however, if this value is higher than
what is set by `.setParallelism()`, it will add to the run-time.

Setter: `.setNumberOfMutationsPerGeneration(<Int>)`

```text
Default: 10
```

###### Genetic Mixing

This setting defines the ratio of impact that the 'best parent' that is used to mutate with a new randomly generated 
child will have upon the mixed-inheritance hyper parameter.  The higher this number, the more effect from the parent the parameter will have.

> Setting this value < 0.1 is effectively using random parameter replacement.  
Conversely, setting the value > 0.9 will not mutate the next generation strongly enough to effectively search the parameter space.

Setter: `.setGeneticMixing(<Double>)`

```text
Default: 0.7

Recommended range: {0.3, 0.8}
```

###### Generational Mutation Strategy

Provides for one of two modes:
* Linear
> This mode will decrease the number of selected hyper parameters that will be mutated each generation.
It is set to utilize the fixed mutation value as a decrement reducer.
        
        Example: 
        
        A model family is selected that has 10 total hyper parameters.
        
        In "linear" mode for the generational mutation strategy, with a fixed mutation value of 1, 
        the number of available mutation parameters for the first mutation generation would be
        set to a maximum value of 9 (randomly selected in range of {1, 9}).
        At generation #2, the maximum mutation count for hyper parameters in the vector space
        will decrememt, leaving a range of randomly selected random hyper parameters of {1, 8}.
        This behavior continues until either the decrement value is 1 or the generations are exhausted.

* Fixed
> This mode sets a static mutation count for each generation.  The setting of Fixed Mutation Value
determines how many of the hyper parameters will be mutated each generation.  There is no decrementing.

Setter: `.setGenerationalMutationStrategy(<Double>)`

```text
Default: "linear"

Available options: "linear" or "fixed"
```

###### Fixed Mutation Value

Setter

```text
Default: 
```

###### Mutation Magnitude Mode

This setting determines the number of hyper parameter values that will be mutated during each mutation iteration.

There are two modes:
* "random"

> In random mode, the setting of `.setGenerationalMutationStrategy()` is used, in conjunction with 
the current generation count, to provide a bounded restriction on the number of hyper parameters
per model configuration that will be mutated.  A Random number of indeces will be selected for
mutation in this range.

* "fixed"

> In fixed mode, a constant count of hyper parameters will be mutated, used in conjunction with 
the setting of .`setGenerationalMutationStrategy()`.
>> i.e. With fixed mode, and a generational mutation strategy of "fixed", each mutation generation
would be static (e.g. fixedMutationValue of 3 would mean that each model of each generation would
always mutate 3 hyper parameters).  Variations of the mixing of these configurations will result in
varying degrees of mutation aggressiveness.

Setter: `.setMutationMagnitudeMode(<String>)`

```text
Default: "fixed"

Available options: "fixed" or "random"
```

###### Auto Stopping Flag

Provides a means, when paired with an auto stopping score threshold value, to early-terminate
a run once a model score result hits the desired threshold.

> NOTE: In batch mode, the entire batch will complete before evaluation of stopping will occur.
There is not a mechanism for stopping immediately upon reaching a successful score.  If 
functionality like this is desired, please use "continuous" mode.

Setter: `.autoStoppingOn()` and `.autoStoppingOff()`

```text
Default: on

Available options: on or off
```

###### Auto Stopping Score

Setting for specifying the early stopping value.  

> NOTE: Ensure that the value specified matches the optimization score set in `.setScoringMetric(<String>)`
>> i.e. if using f1 score for a classification problem, an appropriate early stopping score might be in the range of {0.92, 0.98}

Setter: `.setAutoStoppingScore(<Double>)`

```text
Default: 0.95
```

[WARNING] This value is set as a placeholder for a default classification problem.  If using 
regression, this will ***need to be changed***

### Continuous Mode

Continuous mode uses the concept of micro-batching of hyper parameter tuning, running *n* models
in parallel.  When each Future is returned, the evaluation of its performance is made, compared to
the current best results, and a new hyper parameter run is constructed.  Since this is effectively
a queue/dequeue process utilizing concurrency, there is a certain degree of out-of-order process
and return.  Even if an early stopping criteria is met, the thread pool will await for all committed
Futures to return their values before exiting the parallel concurrent execution context.

Please see the below descriptions about the settings and adhere to the warnings to get optimal performance.

#### Continuous-Specific Configurations

###### Continuous Evolution Max Iterations

This parameter sets the total maximum cap on the number of hyper parameter tuning models that are 
created and set to run.  The higher the value, the better the chances for convergence to optimum 
tuning criteria, at the expense of runtime.

Setter: `.setContinuousEvolutionMaxIterations(<Int>)`

```text
Default: 200
```

###### Continuous Evolution Stopping Score

Setting for early stopping.  When matched with the score type, this target is used to terminate the hyper parameter
tuning run so that when the threshold has been passed, no additional Futures runs will be submitted to the concurrent
queue for parallel processing.

> NOTE: The asynchronous nature of this algorithm will have additional results potentially return after a stopping 
criteria is met, since Futures may have been submitted before the result of a 'winning' run has returned.  
> > This is intentional by design and does not constitute a bug.

Setter: `.setContinuousEvolutionStoppingScore(<Double>)`

**NOTE**: ***This value MUST be overridden in regression problems***

```text
Default: 1.0

This is a placeholder value.  Ensure it is overriden for early stopping to function in classification problems.
```

###### Continuous Evolution Parallelism

This setting defines the number of concurrent Futures that are submitted in continuous mode.  Setting this number too
high (i.e. > 5) will minimize / remove the functionality of continuous processing mode, as it will begin to behave
more like a batch mode operation.

> TIP: **Recommended value range is {2, 5}** to see the greatest exploration benefit of the n-dimensional hyper 
parameter space, with the benefit of run-time optimization by parallel async execution.

Setter: `.setContinuousEvolutionParallelism(<Int>)`

```text
Default: 4
```

###### Continuous Evolution Mutation Aggressiveness

Similar to the batch mode setting `.setFixedMutationValue()`; however, there is no concept of a 'linear' vs 'fixed' 
setting.  There is only a fixed mode for continuous processing.  This sets the number of hyper parameters that will
be mutated during each async model execution.

> The higher the setting of this value, the more the feature space will be explored; however, the longer it may take to 
converge to a 'best' tuned parameter set.

> The recommendation is, for **exploration of a modeling task**, to set this value ***higher***.  If trying to fine-tune a model,
or to automate the **re-tuning of a production model** on a scheduled basis, setting this value ***lower*** is preferred.

Setter: `.setContinousEvolutionMutationAggressiveness(<Int>)`

```text
Default: 3
```

###### Continuous Evolution Genetic Mixing

This mirrors the batch mode genetic mixing parameter.  Refer to description above.

Setter: `.setContinuousEvolutionGeneticMixing(<Double>)`

```text
Default: 0.7

Restricted to range {0, 1}
```

###### Continuous Evolution Rolling Improvement Count

[EXPERIMENTAL]
This is an early stopping criteria that measures the cumulative gain of the score as the job is running.  
If improvements ***stop happening***, then the continuous iteration of tuning will stop to prevent useless continuation.

Setter: `.setContinuousEvolutionRollingImprovementCount(<Int>)`

```text
Default: 20
```

### MLFlow Settings

MLFlow integration in this toolkit allows for logging and tracking of not only the best model returned by a particular run,
but also a tracked history of all hyper parameters, scoring results for validation, and a location path to the actual
model artifacts that are generated for each iteration.

More information: [MLFlow](https://mlflow.org/docs/latest/index.html), [API Docs](https://mlflow.org/docs/latest/java_api/index.html)

The implementation here leverages the JavaAPI and can support both remote and Databricks-hosted MLFlow deployments.

##### Options
```text
member of configuration case class MLFlowConfig()
```

###### MLFlow Logging Flag

Provides for either logging the results of the hyper parameter tuning run to MLFlow or not.

Setters: `.mlFlowLoggingOn()` and `.mlFlowLoggingOff()`

```text
Default: on
```

###### MLFlow Log Artifacts Flag

When in the course of experimentation, it is not neccessary to log the model artifacts of each iteration to MLFlow.
This process of logging artifacts, while useful, is resource-intensive and can lead to long wait times to record all
of the data associated with the run to the tracking server.
However, for CI/CD purposes, as well as re-run-ability, the option is provided to log the artifacts.

Setters: `.mlFlowLogArtifactsOn()` and `.mlFlowLogArtifactsOff()`

```text
Default: off
```

###### MLFlow Tracking URI

Defines the host address for where MLFlow is running.

For ***Databricks hosted MLFlow***, the URI is the shard address (e.g. "https://myshard.cloud.databricks.com")

> This setting ***must be overriden***

Setter: `.setMlFlowTrackingURI(<String>)`

```text
Default: "hosted"

NOTE: This is a placeholder value.
```

###### MLFlow Experiment Name

Defines the name of the experiment run that is being conducted.  
To prevent collisions / overwriting of data in MLFlow, a unique identifier is appended to this String.  It applies to 
all hyper parameter modeling runs that occur within the execution of `.run()`

Setter: `.setMlFlowExperimentName(<String>)`

```text
Default: "default"
```

###### MLFlow API Token

The API Token that allows for access to the MLFlow service.

> In hosted mode (on Databricks), access to this is provided via the dbutils command: `dbutils.notebook.getContext().apiToken.get`
for self-managed deployments, acquire the token securely through non-printable means and pass it to the class through the setter.

Setter: `.setMlFlowAPIToken(<String>)`

```text
Default: "default"

NOTE: This is a placeholder value.  This must be overridden with the correct token.
```

###### MLFlow Model Save Directory

The path root to store all of the models that are generated with each hyper parameter optimization iteration.
These will be preceded and labeled with the UUID that is generated for the run (also logged in mlflow model location parameter)

Setter: `.setMlFlowModelSaveDirectory(<String>)`

```text
Default: "s3://mlflow/experiments/"

NOTE: This is a placeholder and it is HIGHLY RECOMMENDED to not store to an arbitrary bucket named "mlflow".
```

### Model Family Specific Settings

Setters (for all numeric boundaries): `.setNumericBoundaries(<Map[String, Tuple2[Double, Double]]>)`

Setters (for all string boundaries): `.setStringBoundaries(<Map[String, List[String]]>)`

> To override any of the features space exploration constraints, pick the correct Map configuration for the family
that is being used, define the Map values, and override with the common setters.

#### Random Forest

###### Default Numeric Boundaries
```scala
def _rfDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "numTrees" -> Tuple2(50.0, 1000.0),
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "subSamplingRate" -> Tuple2(0.5, 1.0)
  )
```
###### Default String Boundaries
```scala
 def _rfDefaultStringBoundaries = Map(
    "impurity" -> List("gini", "entropy"),
    "featureSubsetStrategy" -> List("auto")
  )
```
#### Gradient Boosted Trees

###### Default Numeric Boundaries
```scala
  def _gbtDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxIter" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "minInstancesPerNode" -> Tuple2(1.0, 50.0),
    "stepSize" -> Tuple2(1E-4, 1.0)
  )
```
###### Default String Boundaries
```scala
  def _gbtDefaultStringBoundaries: Map[String, List[String]] = Map(
    "impurity" -> List("gini", "entropy"),
    "lossType" -> List("logistic")
  )
```
#### Decision Trees

###### Default Numeric Boundaries
```scala
  def _treesDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "maxBins" -> Tuple2(10.0, 100.0),
    "maxDepth" -> Tuple2(2.0, 20.0),
    "minInfoGain" -> Tuple2(0.0, 1.0),
    "minInstancesPerNode" -> Tuple2(1.0, 50.0)
  )
```
###### Default String Boundaries
```scala
  def _treesDefaultStringBoundaries: Map[String, List[String]] = Map(
    "impurity" -> List("gini", "entropy")
  )
```
#### Linear Regression

###### Default Numeric Boundaries
```scala
  def _linearRegressionDefaultNumBoundaries: Map[String, (Double, Double)] = Map (
    "elasticNetParams" -> Tuple2(0.0, 1.0),
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5)
  )
```
###### Default String Boundaries
```scala
  def _linearRegressionDefaultStringBoundaries: Map[String, List[String]] = Map (
    "loss" -> List("squaredError", "huber")
  )
```
#### Logistic Regression 

###### Default Numeric Boundaries
```scala
  def _logisticRegressionDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "elasticNetParams" -> Tuple2(0.0, 1.0),
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5)
  )
```
###### Default String Boundaries
```scala
 def _logisticRegressionDefaultStringBoundaries: Map[String, List[String]] = Map(
    "" -> List("")
  )
```

> NOTE: ***DO NOT OVERRIDE THIS***

#### Multilayer Perceptron Classifier

###### Default Numeric Boundaries
```scala
  def _mlpcDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "layers" -> Tuple2(1.0, 10.0),
    "maxIter" -> Tuple2(10.0, 100.0),
    "stepSize" -> Tuple2(0.01, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5),
    "hiddenLayerSizeAdjust" -> Tuple2(0.0, 50.0)
  )
```
###### Default String Boundaries
```scala
  def _mlpcDefaultStringBoundaries: Map[String, List[String]] = Map(
    "solver" -> List("gd", "l-bfgs")
  )
```
#### Linear Support Vector Machines

###### Default Numeric Boundaries
```scala
  def _svmDefaultNumBoundaries: Map[String, (Double, Double)] = Map(
    "maxIter" -> Tuple2(100.0, 10000.0),
    "regParam" -> Tuple2(0.0, 1.0),
    "tolerance" -> Tuple2(1E-9, 1E-5)
  )
```
###### Default String Boundaries
```scala
  def _svmDefaultStringBoundaries: Map[String, List[String]] = Map(
    "" -> List("")
  )
```
> NOTE: ***DO NOT OVERRIDE THIS***

## Feature Importance

DOCS COMING SOON

## Decision Splits

DOCS COMING SOON



# Inference

## Batch Inference Mode

Batch Inference Mode is a feature that allows for a particular run's settings to be preserved, recalled, and used to 
apply the precise feature engineering actions that occured during a model training run.  This is useful in order to 
utilize the built model for batch inference, or to understand how to write real-time transformation logic in order to 
create a feature vector that can be sent through a model in a model-as-a-service mode.

***This mode REQUIRES MLFlow in order to function***

Each individual model will get two forms of the configuration:
1. A compact JSON data structure that is logged as a tagged element to the mlflow run that can be copied and 
pasted (or retrieved through the mlflow API) for any particular run that would be used to do batch inference
2. A specified location that a Dataframe has been saved that contains the same location.

There are two primary entry points for an inference run.  One simply requires the path of the DataFrame (preferred), 
while the other requires the json string itself.  Everything needed to execute the inference prediction is 
contained within this data structure (either the json or the Dataframe)

### Pipeline API
To use pipeline API, check [here](PIPELINE_API_DOCS.md) for an example usage.

