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
The python APIs currently support the three following classes:
1. `FeatureImportance`
2. `AutomationRunner`
3. `FamilyRunner`


### Feature Importance Class
For more information about the underlying algorithms please see [APIDOCS](https://github.com/databricks/providentia/blob/master/APIDOCS.md#automl-toolkit)
This class takes six parameters for instantiation:

`model_family` - one of the supported model families listed [HERE](https://github.com/databricks/providentia/blob/master/APIDOCS.md#automl-toolkit)


`prediction_type` - either "regressor" or "classifier"


`df` - Dataframe that will be used for feature importance algorithm

`cutoff_value` - threshold value for feature importance algorithm

`cutoff_type` - cutoff for the feature algorithm 

`overrides` - dictionary of overrides for feature importance configuration

Below is an example of using the `FeatureImportance` class on Databricks:
```python

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
from py_auto_ml.exploration.feature_importance import FeatureImportance

fi_importances_package = FeatureImportance("XGBoost", "classifier",  source_data,20.0,"count",generic_overrides)
```
Once the feature importance algorithm has been run, there are two dataframes that remain as attributes of the instance 
of the class. The first is the `importances` dataframe which lists the features and their importance value. The second
is the `top_fields` dataframe which consists only of the features themselves. Below is an example of retrieving these
dataframes once the feature importance algorithm has been run.

```python
##Retrieving the importances DF
fi_importances.importances.show()

## Retrieving the top_fields DF
fi_importances.top_fields.show()
``` 

### AutomationRunner Class
The `AutomationRunner` class invokes the `automationRunner` Scala library via the JVM. This class has a few different
type of runs you can read more about [HERE](https://github.com/databricks/providentia/blob/master/APIDOCS.md#full-automation). To call the class you can pass it a 

`model_family` - one of the supported model families listed HERE

`prediction_type` - either "regressor" or "classifier"

`data_frame` - Dataframe that will be used for feature importance algorithm

`runner_type` - either "run", "confuison", or "prediction"

`overrides` - dictionary of configuration overrides. If null, this will run with default configurations

Below is an example of calling the `AutomationRunner` class with the overrides defined above
```python
model_family = "XGBoost"
prediction_type = "classifier"
run_type = "confusion"

## Kickoff Automation runner
from py_auto_ml.automation_runner import AutomationRunner

runner = AutomationRunner(model_family,
                         prediction_type,
                         source_data,
                         run_type,
                         generic_overrides)
```

Based on the `run_type`, the object will have different attributes. 

| Run Type     | Attributes                                                       |
|--------------|------------------------------------------------------------------|
| "run"        | generation_report, model_report                                  |
| "confusion"  | confusion_data, prediction_data, generation_report, model_report |
| "prediction" | data_with_predictions, generation_report, model_report           |


### Family Runner 
The `FamilyRunner` class kicks of the `familyRunner` equivalent in the scala library. This allows the user to run 
several different model families in parallel. The `FamilyRunner` class takes three necessary parameters:

`family_configs` - a dictionary that contains the `model_family` as the key and a dictionary of overrides as the value

`prediction_type` - either `regressor` or `classifier`

`df` - Spark Dataframe 

Below is an example of calling the `FamilyRunner` class:
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
  "mlFlowModelSaveDirectory": "/dbfs/tmp/mgm/ml",
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
  "mlFlowModelSaveDirectory": "/dbfs/tmp/mgm/ml",
  "pipelineDebugFlag": True
}
# Import the family runner
from py_auto_ml.executor.family_runner import FamilyRunner

prediction_type = "classifier"
family_runner_configs = {
  "XGBoost": xg_boost_overrides,
  "LogisticRegression": logisticRegOverrides
}

family_runner = FamilyRunner(family_runner_configs,
                            prediction_type,
                            source_data)

```

The instance of the `FamilyRunner` class currently has a few attribures:
1. `model_report`
2. `generation_report`
3. `best_mlflow_run_id`

### Features in Current Development
- Support for tuning Algoritm Configs 
- Full pipeline API support
