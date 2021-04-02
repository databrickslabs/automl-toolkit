## Auto ML Toolkit Release Notes

### Version 0.8.1
#### Features
* Added Distributed Shapley calculation APIs :
```scala 
com.databricks.labs.automl.exploration.analysis.shap.ShapleyPipeline
com.databricks.labs.automl.exploration.analysis.shap.ShapleyModel
```
for calcualting shap values for each feature within a trained model (or pipeline)'s feature vector.
see: [Documentation](ANALYSIS_TOOLS_DOCS.md) for details on the new API.

* Added Tree-based SparkML model and metrics visualizations and extractors:
```scala
com.databricks.labs.automl.exploration.analysis.trees.TreeModelVisualization
com.databricks.labs.automl.exploration.analysis.trees.TreePipelineVisualization
```

Update the group ID to relfect the updated Databricks Labs naming convention. See the project definitions in [pom.xml](pom.xml) or [build.sbt](build.sbt).

see [Docs](ANALYSIS_TOOLS_DOCS.md) for API details.

### Version 0.7.1
#### Features
* Complete overhaul of train/test splitting and kFolding.  Prior to this performance scaling improvement, 
the train and test data sets would be calculated during each model's kfold stage, resulting in non-homogenous
comparisons between hyper parameters, as well as performance degradation from consantly having to perform 
splitting of the data.  There are three new parameters to control the behavior of splitting:
```text
Configuration parameters:

- "tunerDeltaCacheBackingDirectory"
  This new setting will define a location on dbfs in order to write extremely large training and test split data sets
  to.  This is particularly recommended for use cases in which the volume of the data is so large that making even a 
  few copies of the raw data would exceed budgeted cluster size allowances (recommended for data sets that are in the 
  100's of GB range - TB range)
- "tunerDeltaCacheBackingDirectoryRemovalFlag"
  This new setting (Boolean flag) will determine, if using 'delta' mode on splitCachingStrategy, whether or not
  to delete the delta data sets on dbfs after training is completed.  By default, this is set to true (will delete and 
  clean up the directories).  If further evaluation or testing of the train test splits is needed after the run is 
  completed, or if investigation into the composition of the splits is desired, or if auditing of the training data
  is required due to business rules, set this flag to false.  
  NOTE: directory pathing prevention of collisions is done through the generation of a UUID.  The path on dbfs
  will have this as part of the root bucket for the run.
- "splitCachingStrategy" DEFAULT: 'persist'
  Options: 'cache', 'persist', or 'delta'
  - delta mode: this will perform a train/test split for each kFold specified in the job definition, write the train
  and test datasets to dbfs in delta format, and provide a reference to the delta source for the training run.  
    NOTE: this will incur overhead and is NOT recommended for data sets that can easily fit multiple copies into 
    memory on the cluster.  
  - persist mode: this will cache and persist the train and test kFold data sets onto local disk.  This is recommended
  for larger data sets that in order to fit n copies of the data in memory would require an extremely large or 
  expensive cluster.  This is the default mode.
  - cache mode: this will use standard caching (memory and disk) for the kFold sets of train and test.  This mode
  is only recommended if the data set is relatively small and k copies of the data set can comfortably reside in memory
  on the cluster.
```
* Main config is now written and tracked via MLFlow. Any pipeline trained as of 0.7.1 will provide a full config
in json format in MLFlow Artifacts and next to your saved models path.

* Run Inference Pipelines with only a RunID. You no longer have to track and manage a LoggingConfig to pass into
the inference pipeline. That constructor has been deprecated, only use it for legacy pipelines. Old training pipelines
will not be able to run this way but all future pipelines created as of 0.7.1 will be able to run with only the 
MLFlow runId.

#### Bug Fixes / Improvements
* scoring metric can now support resolution of differently spelled metrics (upper case, camel case, etc.)
  and will resolve to the standard naming conventions within SparkML for Binary, Multiclass, and Regression
  evaluators.
* Model training was getting one additional fold than applied at configuration, this has been resolved.
* Type casting enabled from python API for complex nested types to config
* Minor changes to assertions to provide a better experience
* Minor internal function changes

### Version 0.6.2
#### Features
* Added support for PySpark!  There is now a Python API for Databricks labs automl!
* Added FeatureInteraction module (configurable to interact either all features or only those that pass checks for 
perceived gain of adding the interacted feature based on its parents)
```text
Configuration of feature interaction modes is through setting the configurations:

- featureInteractionFlag -> true (turns the feature on.  Default: false)
- featureInteractionRetentionMode -> one of: 'all', 'optimistic', or 'strict'
    (all -> interacts all features and will include them in the feature vector
     optimistic -> compares each interacted column to it's parents and if it is at least
     it is at least 1 - featureInteractionTargetInteractionPercentage as good as EITHER parent
     it will be retained as a feature.
     strict -> it must be 1 - featureInteractionTargetInteractionPercentage as good as 
     BOTH parents to be included.
    )
- featureInteractionContinuousDiscretizerBucketCount -> Default 10 (sets the number of quantization buckets to use when
    handling continuous features in order to properly calculate InformationGain for Classification Models.  Increasing 
    this value may provide greater accuracy at the cost of runtime performance)
- featureInteractionParallelism -> Default 12 (the interacted features are created and scored 
    asynchronously.  This parallelism setting is separated from the other two parallelism 
    values due to the distinctly different level of CPU consumption that is required to perform this
    stage.  Overriding the default is recommended and is intended to be set in accordance with 
    the size of the cluster executing the run.)
- featureInteractionTargetInteractionPercentage -> Default 10.0 (provides the threshold for the 
    retention modes 'strict' and 'optimistic' for determining whether to keep an interacted column based on the 
    relative percentage difference between the parents of the interacted column and the interacted column.
    It is measured as a "must be at least 1 - x% as good at y" wherein x is the percentage to be included and  y is 
    either Variance or Information Gain.  i.e. : with this value set to 10 for a classification problem, 
    an interacted column would be included in the feature vector if it's Information Gain was greater than 
    or equal to 80% of the Information Gain of its parents)
```
* Added the ability to calculate Differential Entropy for Regression Tasks (supported in both FeatureInteraction and 
in the new Pearson Filtering algorithm)
* Full refactor of Pearson Filtering and Feature Correlation to utilize DataFrames for column-wise comparisons 
 Adjusted core looping algorithm to support exactly 1:1 checking of n x m column validation.  Speed improvements are 
 dramatic for data sets utilizing higher numbers of features.
 * Improved cardinality detection for data types and inspection of correlation detection based on approrpriate 
 methodology (Information Gain / Entropy for Classifiers, Differential Entropy for Regressors) and handling of 
 nominal vs continuous numeric types correctly for such validators.
 
#### Bug Fixes
* Adjusted Pipeline OneHotEncoder to ensure prevention of metadata loss from StringIndexers for inference .transforms()
    through the use of additional StringIndexer stages immediately preceding the OneHotEncoder stages.
* Stability improvements - creation of ~63 new unit tests to validate core functionality
* Over 200 issues solved due to new unit testing test suite (not going to mention them all here)

### Version 0.6.1
* Upgraded MlFLow to 1.3.0
* Pipeline now registers with Mlflow (including Inference Pipeline Model and feature engineered original df)
* Added new Pipeline APIs to Run inference directly against MLFlow Run Ids
* Training Pipeline now automatically registers pipeline progress and each stages transformations with MLFlow

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
import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
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