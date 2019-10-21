## Auto ML Toolkit Release Notes

### Version 0.6.0

#### Features
* New APIs around Spark ML pipeline semantics for fetching full inference as well as feature engineering pipelines. See [this](PIPELINE_API_DOCS.md) for the usage 
* MainConfig settings are now pretty printed to stdout and logged as json strings to aid in readability.
* PostModelingOptimization will now search through a logspace based on euclidean distance of vector similarity to 
minimize (not remove) the probability of too-similar hyper parameters from being tested in final phase.
[NOTE] - this feature is not supported for MLPC due to the complexity involved in layer estimation 
for distance calculations.
* MLflow settings are now defaulted: api Key, uri are default configured to work with the current notebook context
that is calling the class.  These can still be overridden.  
* MLflow logging is now defaulted to the same parent directory of the notebook executing it through reflection
during runtime.  This is to maintain parity with how hosted MLFlow works.  This can be overridden if an alternate
Workspace path is desired for logging to.
* Binary Encoder stand-alone package (transformer) has been added and is compatible with the SparkML Pipeline API
This is intended to be used as an alternative to OneHotEncoding for high cardinality
nominal fields.  It is an information loss algorithm, though.  Integration options with the automl
toolkit will be coming in a future release.
* Added a new MBO algorithm on top of the Genetic Algorithm.  In each generation, a larger count of potential
candidates are now generated, which are then fed, along with apriori hyperparameter + score information to a 
new package (GenerationOptimizer.scala) which will train a Regressor (selectable) and return the best predicted
hyperparameter combinations.
* Added the following additional configuration options:
```text
dataPrepParallelism -> allows for setting a separate parallelism factor for the feature engineering phase
of the application (can be useful for extremely large data sets to have a lower parallelism value than the 
tuning parallelism setting)

tunerGeneticMBOCandidateFactor -> Integer that serves as a multiplicative adjustment to the number of candidates
that are mutated and generated from each genetic mutation epoch (only applies to stages other than first and last)

tunerGeneticMBORegressorType -> One of "XGBoost", "RandomForest", or "LinearRegression"

tunerContinuousEvolutionImprovementThreshold -> allows for an additional stopping criteria based on cumulative 
gains of improvement.  NOTE: must be negative and values less negative than -5 will likely cause early stopping
in continuous mode if parallelism is set too high.  Adjust to values closer to 0 than -5 with caution!

```

#### Bug Fixes
* If a setter(s) were used after the mainConfig was set on AutomationRunner, the default values would be applied
to the instance of the Automation or FamilyRunner.  This behavior has been fixed and chained setters can be used
even after the mapped configuration has been applied.
* KSample (distributed SMOTE) bug fixes for scalability and reliability.
* Eliminated the scaling bug when using a model that doesn't have ksample as its trainSplitMethodology set has a 
scaling task set.
* enabled asynchronous support for variance filtering to reflect the dataPrepParallelism setting (was hard-coded before to 10)
* changed default logging location for mlflow to support azure shards


### Version 0.5.2

#### Fix XGBoost 0.9.0 issue with classifiers
XGBoost implementation for Spark has a default override for missing value imputation that is not compatible with SparkML.
Modifying the default behavior of XGBoost allows the new version to work correctly.
#### Adding support for naFill manual override:
##### New Modes: 
* <i>"auto"</i> - previously the only mode (uses statistical options to infer missing data) Usage of .setNumericFillStat and .setCharacterFillStat will inform the type of statistics that will be used to fill.
* <i>"mapFill"</i> - Custom by-column overrides.  Column names specified in either of the two maps (numericNAFillMap and/or categoricalNAFillMap) MUST be present in the DataFrame schema.  
    All fields not 	specified in these maps will use the stastics-based approach to fill na's.
* <i>"blanketFillAll"</i> - Fills na's throughout the DataFrame with values specified in characterNABlanketFillValue and numericNABlanketFillValue.
* <i>"blanketFillCharOnly"</i> - will use the characterNABlanketFillValue for only categorical columns, while the stats method will be used for numerics.
* <i>"blanketFillNumOnly"</i> - will use the numericNABlanketFillValue for only categorical columns, while the stats method will be used for character columns.
######Example:
```scala
val overrides = Map(
  "labelCol" -> "myLabel",
  "fillConfigCategoricalNAFillMap" -> Map("native_country" -> "us"),
  "fillConfigNumericNAFillMap" -> Map("education_years" -> 12.0, "capital_loss" -> 0.0),
  "fillConfigCharacterNABlanketFillValue" -> "missing",
  "fillConfigNumericNABlanketFillValue" -> 0.0,
  "fillConfigNAFillMode" -> "mapFill"
)
```
#### Safety Checks for High Cardinality Non-numeric columns
To prevent against extreme feature vector size 'explosion' with improperly defined high cardinality feature fields (i.e. a userID or telephone number is in the data set), cardinality checks are now enabled during feature vector creation.  The behavior of these is controlled with the following setters:
- <i>setCardinalityCheck</i> (default true) - disables or enables this feature
- <i>setCardinalityCheckMode</i> - either "silent" or "warn"
    * <u>Silent mode</u>: Removes high cardinality non-numeric fields from the feature vector.
    * <u>Warn mode</u>: If a non-numeric field is found that is above the specified threshold, an exception is thrown.
- <i>setCardinalityLimit</i> Integer limit above which the field will be removed (silent mode) or an exception will be thrown (warn mode)
- <i>setCardinalityPrecision</i> Optional override to the precision for checking the approx distinct cardinality nature of a non-numeric field.
- <i>setCardinalityType</i> Whether to use distinct or approx_distinct (approx_distinct is <i><b>highly recommended</b></i> for large data sets)
######Example:
```scala
val overrides = Map(
  "labelCol" -> "myLabel",
  "fillConfigCardinalitySwitch" -> true,
  "fillConfigCardinalityType" -> "exact",
  "fillConfigCardinalityPrecision" -> 0.9,
  "fillConfigCardinalityCheckMode" -> "warn",
  "fillConfigCardinalityLimit" -> 100
)
```
#### KSampling (Distributed SMOTE)
Now supported in the autoML Toolkit is 'intelligent minority class oversampling'. 
- This is a new train / test split method for classification problems with heavy class imbalance.
- To use, simply specify in the configuration map:
######Example:

```scala
val overrides = Map(
  "labelCol" -> "myLabel",
  "tunerTrainSplitMethod" -> "kSample",
  "tunerKSampleSyntheticCol" -> "synth_KSample",
  "tunerKSampleKGroups" -> 25,
  "tunerKSampleKMeansMaxIter" -> 200,
  "tunerKSampleKMeansTolerance" -> 1E-6,
  "tunerKSampleKMeansDistanceMeasurement" -> "euclidean",
  "tunerKSampleKMeansSeed" -> 42L,
  "tunerKSampleKMeansPredictionCol" -> "kGroup_sample",
  "tunerKSampleLSHHashTables" -> 10,
  "tunerKSampleLSHSeed" -> 42L,
  "tunerKSampleLSHOutputCol" -> "hashes_ksample",
  "tunerKSampleQuorumCount" -> 7,
  "tunerKSampleMinimumVectorCountToMutate" -> 1,
  "tunerKSampleVectorMutationMethod" -> "random",
  "tunerKSampleMutationMode" -> "weighted",
  "tunerKSampleMutationValue" -> 0.5,
  "tunerKSampleLabelBalanceMode" -> "target",
  "tunerKSampleCardinalityThreshold" -> 20,
  "tunerKSampleNumericRatio" -> 0.2,
  "tunerKSampleNumericTarget" -> 10000
)

val myConfig = ConfigurationGenerator.generateConfigFromMap("XGBoost", "classifier", overrides)

val runner = FamilyRunner(data, Array(myConfig)).execute

```
- Before turning on this mode, ensure that:
  * Modeling type is of 'classfication'
  * The modes "target" and "percentage" are subject to RunTime checks.  If the numeric ratio or target (for the 
  respective mode selected) are, for the smallest minority class count, bigger than these target values, a RunTimeException
  will be thrown.  
  
Further details of the implementation, performance, and usage of this new feature will be extensively documented in an upcoming blog post by Databricks.