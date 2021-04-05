# AutoML-Toolkit

The AutoML-Toolkit is an automated ML solution for Apache Spark. It provides common data cleansing and feature 
engineering support, automated hyper-parameter tuning through distributed genetic algorithms, and model tracking 
integration with MLFlow. It currently supports Supervised Learning algorithms that are provided as part of Spark Mllib.

The python APIs are a means towards interfacing with the Scala library via pyspark. 

## Setup 
Currently, this library exists as a `.whl` file in the `/dist` directory. You can also run the following in a terminal 
to build the wheel locally:
```
python setup.py bdist_wheel
```

## General Overview 
The python APIs currently support the four following classes:
1. `FeatureImportance`
2. `AutomationRunner`
3. `FamilyRunner`
4. `Shapley`


### Feature Importance Class
For more information about the underlying algorithms please see [APIDOCS](https://github.com/databricks/providentia/blob/master/APIDOCS.md#automl-toolkit). 
Feature importances are run via the `run_feature_importace` function within an instance of the `FeatureImportance` 
class.

`model_family` - one of the supported model families listed [Here](https://github.com/databricks/providentia/blob/master/APIDOCS.md#automl-toolkit)


`prediction_type` - either "regressor" or "classifier"


`dataframe` - Dataframe that will be used for feature importance algorithm

`cutoff_value` - threshold value for feature importance algorithm

`cutoff_type` - cutoff for the feature algorithm 

`overrides` - dictionary of overrides for feature importance configuration

Below is an example of using the `FeatureImportance` class on Databricks:
```python

source_data = spark.read.parquet("/tmp/loan-risk-analysis/loan-risk-analysis-full-cleansed.parquet").withColumnRenamed("bad_loan", "label")

## Generic configuration
experimentNamePrefix = "/Users/marygrace.moesta@databricks.com/AutoML"
RUNVERSION = "5"
labelColumn = "label"
runExperiment = "runRF_" + RUNVERSION
projectName = "mg_AutoML_Demo"
modelSaveFolder = "/tmp/mgm/ml/automl/"

## This is the configuration of the hardware available (default of 4, 4, and 4)
nodeCount = 8
coresPerNode = 16
totalCores = nodeCount * coresPerNode
driverCores = 30

## Save locations
mlFlowModelSaveDirectory = "dbfs:" + modelSaveFolder + "models/" + projectName + "/"
inferenceConfigSaveLocation = "dbfs:" + modelSaveFolder + "inference/" + projectName + "/"
cntx = dbutils.entry_point.getDbutils().notebook().getContext()
api_token = cntx.apiToken().get()
api_url = cntx.apiUrl().get()
notebook_path = cntx.notebookPath().get()
generic_overrides = {
  "labelCol": labelColumn,
  "scoringMetric": "areaUnderROC",
  "dataPrepCachingFlag": False,
  "autoStoppingFlag": True,            
  "tunerAutoStoppingScore": 0.91,
  "tunerParallelism": driverCores,
  "tunerKFold": 1,  ## normally should be >=5
  "tunerSeed": 42,  ## for reproducibility
  "tunerInitialGenerationArraySeed": 42,
  "tunerTrainPortion": 0.7,
  "tunerTrainSplitMethod": "stratified",
  "tunerInitialGenerationMode": "permutations",
  "tunerInitialGenerationPermutationCount": 8,
  "tunerInitialGenerationIndexMixingMode": "linear",
  "tunerFirstGenerationGenePool": 16,
  "tunerNumberOfGenerations": 3,
  "tunerNumberOfParentsToRetain": 2,
  "tunerNumberOfMutationsPerGeneration": 4,
  "tunerGeneticMixing": 0.8,
  "tunerGenerationalMutationStrategy": "fixed",
  "tunerEvolutionStrategy": "batch",
  "tunerHyperSpaceInferenceFlag": True,
  "tunerHyperSpaceInferenceCount": 400000,
  "tunerHyperSpaceModelType": "XGBoost",
  "tunerHyperSpaceModelCount": 8,
  "mlFlowLoggingFlag": True,
  "mlFlowLogArtifactsFlag": False,
  "mlFlowTrackingURI": api_url,
  "mlFlowExperimentName": experimentNamePrefix +"/" + projectName+ "/" + runExperiment,
  "mlFlowAPIToken": api_token,
  "mlFlowModelSaveDirectory": mlFlowModelSaveDirectory,
  "mlFlowLoggingMode": "bestOnly",
  "mlFlowBestSuffix": "_best",
  "inferenceConfigSaveLocation": inferenceConfigSaveLocation
  }
  
  ## Calculate Feature Importance 
from databricks.labs.automl_toolkit.exploration.feature_importance import FeatureImportance

FI = FeatureImportance()

fi_importances = FI.run_feature_importances("XGBoost", "classifier",  dataframe,20.0,"count",generic_overrides)
```
Once the feature importance algorithm has been run, there are two dataframes that remain as attributes of the instance 
of the class. The first is the `importances` dataframe which lists the features and their importance value. The second
is the `top_fields` dataframe which consists only of the features themselves. Below is an example of retrieving these
dataframes once the feature importance algorithm has been run.

```python
##Retrieving the importances DF
fi_importances['importances']

## Retrieving the top_fields DF
fi_importances['top_fields']
``` 

### AutomationRunner Class
The `run_automation_runner` function invokes the `runAutomationRunner` Scala library via the JVM. This class has a few different
type of runs you can read more about [Here](https://github.com/databricks/providentia/blob/master/APIDOCS.md#full-automation). To call the function you can pass it a 

`model_family` - one of the supported model families listed HERE

`prediction_type` - either "regressor" or "classifier"

`data_frame` - Dataframe that will be used for feature importance algorithm

`runner_type` - either "run", "confusion", or "prediction"

`overrides` - dictionary of configuration overrides. If null, this will run with default configurations

Below is an example of calling the `run_automation_runner` function with the overrides defined above
```python
## Bring in the dataset 
from pyspark.sql.functions import col,expr, when 
dataframe = spark.read.parquet("/tmp/loan-risk-analysis/loan-risk-analysis-full-cleansed.parquet")\
  .withColumn("label", when((col("bad_loan") == "true"), 1).otherwise(0))\
  .drop(col("bad_loan"))\
  .drop(col("net"))\
  .sample(False, 0.025, 42)\
  .repartition(192)


#Splitting Train and Test
dataset_train = dataframe.where(expr("issue_year <= 2015")).cache()
dataset_valid = dataframe.where(expr("issue_year > 2015")).cache()
dataset_train.createOrReplaceTempView("dataset_train")
dataset_valid.createOrReplaceTempView("dataset_valid")

model_family = "XGBoost"
prediction_type = "classifier"
run_type = "confusion"

## Kickoff Automation runner
from databricks.labs.automl_toolkit.automation_runner import AutomationRunner


runner = AutomationRunner.run_automation_runner(model_family,
                         prediction_type,
                         dataframe,
                         run_type,
                         generic_overrides)
```

Based on the `run_type`, the object will return a dictionary of the following dataframes: 

| Run Type     | Attributes                                                       |
|--------------|------------------------------------------------------------------|
| "run"        | generation_report, model_report                                  |
| "confusion"  | confusion_data, prediction_data, generation_report, model_report |
| "prediction" | data_with_predictions, generation_report, model_report           |


### Family Runner 
The `run_family_runner`function that lives within the `FamilyRunner` class kicks of the `runFamilyRunner` equivalent in 
the scala library. This allows the user to run 
several different model families in parallel. The `run_family_runner` class takes three necessary parameters:

`dataframe` - Spark Dataframe

`prediction_type` - either `regressor` or `classifier`

`family_configs` - a dictionary that contains the 

`model_family` as the key and a dictionary of overrides as the value




Below is an example of calling the `run_family_runner` function:
```python
## Generic configuration
experimentNamePrefix = "/Users/marygrace.moesta@databricks.com/AutoML"
RUNVERSION = "1"
labelColumn = "label"
xgBoostExperiment = "runXG_" + RUNVERSION
logisticRegExperiment = "runLG_" + RUNVERSION
projectName = "MGM_AutoML_Demo"

## This is the configuration of the hardware available
nodeCount = 4
coresPerNode = 4
totalCores = nodeCount * coresPerNode
driverCores = 4

cntx = dbutils.entry_point.getDbutils().notebook().getContext()
api_token = cntx.apiToken().get()
api_url = cntx.apiUrl().get()
notebook_path = cntx.notebookPath().get()
xg_boost_overrides = {
  "labelCol": labelColumn,
  "scoringMetric": "areaUnderROC",
  "oneHotEncodeFlag": True,
  "autoStoppingFlag": True,
  "tunerAutoStoppingScore" : 0.91,
  "tunerParallelism" : driverCores * 2,
  "tunerKFold" : 2,
  "tunerTrainPortion": 0.7,
  "tunerTrainSplitMethod": "stratified",
  "tunerInitialGenerationMode": "permutations",
  "tunerInitialGenerationPermutationCount": 8,
  "tunerInitialGenerationIndexMixingMode": "linear",
  "tunerInitialGenerationArraySeed": 42,
  "tunerFirstGenerationGenePool": 16,
  "tunerNumberOfGenerations": 3,
  "tunerNumberOfParentsToRetain": 2,
  "tunerNumberOfMutationsPerGeneration": 4,
  "tunerGeneticMixing": 0.8,
  "tunerGenerationalMutationStrategy": "fixed",
  "tunerEvolutionStrategy": "batch",
  "tunerHyperSpaceInferenceFlag": True,
  "tunerHyperSpaceInferenceCount": 400000,
  "tunerHyperSpaceModelType": "XGBoost",
  "tunerHyperSpaceModelCount": 8,
  "mlFlowLoggingFlag": True,
  "mlFlowLogArtifactsFlag": False,
  "mlFlowTrackingURI": api_url,
  "mlFlowExperimentName": experimentNamePrefix +"/" + projectName+ "/" + xgBoostExperiment,
  "mlFlowAPIToken": api_token,
  "mlFlowLoggingMode": "bestOnly",
  "mlFlowBestSuffix": "_best",
  "mlFlowModelSaveDirectory": "dbfs:/tmp/mgm/ml",
  "pipelineDebugFlag": True
}

logisticRegOverrides = {
  "labelCol": labelColumn,
  "scoringMetric" : "areaUnderROC",
  "oneHotEncodeFlag": True,
  "autoStoppingFlag": True,
  "tunerAutoStoppingScore": 0.91,
  "tunerParallelism": driverCores * 2,
  "tunerKFold": 2,
  "tunerTrainPortion": 0.7,
  "tunerTrainSplitMethod": "stratified",
  "tunerInitialGenerationMode": "permutations",
  "tunerInitialGenerationPermutationCount": 8,
  "tunerInitialGenerationIndexMixingMode": "linear",
  "tunerInitialGenerationArraySeed": 42,
  "tunerFirstGenerationGenePool": 16,
  "tunerNumberOfGenerations": 3,
  "tunerNumberOfParentsToRetain": 2,
  "tunerNumberOfMutationsPerGeneration": 4,
  "tunerGeneticMixing": 0.8,
  "tunerGenerationalMutationStrategy": "fixed",
  "tunerEvolutionStrategy": "batch",
  "mlFlowLoggingFlag": True,
  "mlFlowLogArtifactsFlag": False,
  "mlFlowTrackingURI": api_url,
  "mlFlowExperimentName": experimentNamePrefix +"/" + projectName+ "/" + logisticRegExperiment,
  "mlFlowAPIToken": api_token,
  "mlFlowLoggingMode": "bestOnly",
  "mlFlowBestSuffix" : "_best",
  "mlFlowModelSaveDirectory": "dbfs:/tmp/mgm/ml",
  "pipelineDebugFlag": True
}
# Import the family runner
from databricks.labs.automl_toolkit.executor.family_runner import FamilyRunner

family_runner = FamilyRunner()
prediction_type = "classifier"
family_runner_configs = {
  "XGBoost": xg_boost_overrides,
  "LogisticRegression": logisticRegOverrides
}

family_runner = family_runner.run_family_runner(dataframe,
                                                prediction_type,
                                                family_runner_configs)

```

The return of the 'run_family_runner' function is a dictionary of the following dataframes:
1. `model_report`
2. `generation_report`
3. `best_mlflow_run_id`

## Using the Family Runner for Inference
There is currently support for the pipeline api in pyspark. There are two ways to run inference on a modeling pipeline:
1. By MLflow run id
2. Pipeline model path

The `mlflow_pipeline_inference` function, that lives in the `FamilyRunner` class, takes the following parametersB;

`run_id` - the mlflow run_id of interest

`model_family` - a supported model family

`prediction_type` - either `regressor` or `classifier`

`datafrme` - a pyspark dataframe that will be used for inference 

`configs` - a dictonary of configs to override default values

`label` - the name of the label column, i.e. the column the model is predicting

Below is an example of running a full inference pipeline via the mlflow run_id from the `family_runner` above
```python
mlflow_inference_df = family_runner.mlflow_pipeline_inference(run_id,
                               "XGBoost",
                               "classifier",
                               source_data,
                               xg_boost_overrides,
                               "label")
```
The `mlflow_pipeline_inference` function returns a dataframe that includes the original dataframe used for inferece plus
additional columns with the feature vector, raw prediction, probability (if applicable), and the prediction. 

Inference can be run directly against the patch of a pipeline model already created by the AutoML Toolkit. The 
`path_pipeline_inference` function (which lives in the `FamilyRunner` class) takes the following parameters:

`path` - the path of the pipelined model 

`dataframe` - a pyspark dataframe that will be used for inference 

Below is an example of running inference directly against the path for a pipelined model created by AutoML:
```python
pipeline_save_path = "/dbfs/tmp/mgm/ml/BestRunclassifier_XGBoost_862a8ceacb534404b86f8bdae69c6449/BestPipeline"
path_df = family_runner.path_pipeline_inference(pipeline_save_path,
                                                    source_data)
```
The `path_pipeline_inference` function returns a dataframe that include the original datafrmae passed as an argument 
plus additional columns with the feature vector, raw prediction, probability (if applicable), and the prediction 

The Family Runner APIs can also be used for feature engineering tasks based on a selected number of configs. The 
`feaure_eng_pipeline` function (which lives in the `FamilyRunner` class) takes the following parameters:

`dataframe` - a pyspark dataframe that will be feature engineered

`model_family` - a supported model family

`prediction_type` - either `regressor` or `classifier`

`configs` - the set of configs used for the Family Runner

Below is 
an example of using the family runner to generate a feature engineered dataframe:
```python
feat_eng_df = family_runner.feature_eng_pipeline(source_data,
                                                  "XGBoost",
                                                  "classifier",
                                                  family_runner_configs)
```
The `feature_eng_pipeline`function returns a feature engineered dataframe base on the 

### Shapley 

For more details see the [AnalysisTools](https://github.com/databricks/providentia/blob/master/ANALYSIS_TOOLS_DOCS.md)

```python
from pyspark.ml.regression import LinearRegressionModel

from databricks.labs.automl_toolkit.exploration.shapley import Shapley

## Load a pre-trained LinearRegression Model
model_path = "dbfs:/Users/nick.senno/shap/models/boston-linear/"
model = LinearRegressionModel.load(model_path)

## Load pre-scored data set used to train model 
feature_cols = ["INDUS", "LSTAT", "DIS"]
cols = feature_cols + ["features"] + ["label"]
dataDF = spark.table("shap.boston_processed")
featuresDF = dataDF.select("features")

shapley = Shapley(featuresDF, model, "features", 10, 1000, 1621)

## compute per record shapley values with approximate error 
shapley_results = shapley.calculate()

feature_results = shapley.feature_aggregated_shap(feature_cols)

```

The Shapley class takes the following parameters: 
`feature_data` the Spark DataFrame that was used during model training (with feature vector) and result (label vector)
`model`  PySpark model 
`feature_col` the name of the single feature column created by the PySpark `VectorAssembler`
`repartition_value` degree to which the training data is split. Higher numbers equate to more conncurrent SHAP calculations per partition but less data per partition
`vector_mutations` number of vector mutations to consider per each record 
`random_seed` seed for random number generator for creating vector mutations


To calculate the aggregated Shapley values for each feature, use the `feature_aggregated_shap` method which takes a list of the original feature column names
