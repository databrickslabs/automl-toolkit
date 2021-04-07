# Databricks Labs AutoML Toolkit
[Release Notes](RELEASE_NOTES.md) |
[Python API Docs](python/docs/APIDOCs.md) |
[Python Artifact](python/dist/pyAutoML-0.2.0-py3-none-any.whl) |
[Developer Docs](APIDOCS.md) |
[Python Docs](python/docs/APIDOCs.md) |
[Analysis Tools Docs](ANALYSIS_TOOLS_DOCS.md) |
[Demo](demos) |
[Release Artifacts](bin) |
[Contributors](#core-contribution-team)


This Databricks Labs project is a non-officially-supported end-to-end supervised learning solution for automating:
* Feature clean-up 
    * Advanced NA fill, covariance calculations, collinearity determination, outlier filtering, and data casting
* Feature Importance calculation suite
    * RandomForest or XGBoost determinations
* Feature Interaction with [Information Gain selection](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
* Feature vectorization
* Advanced train/test split techniques (including Distributed [SMOTE](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis#SMOTE) (KSample))
* Model selection and training
* Hyper parameter optimization and selection
    * Hyperspace, Genetic, and MBO-based selection
* Batch Prediction through serialized [SparkML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html)
* Logging of model results and training runs (using [MLFlow](https://mlflow.org))
* Model interprability (including distributed [Shapley Values](https://christophm.github.io/interpretable-ml-book/shapley.html) )

This package utilizes Apache Spark ML and currently supports the following model family types:

* Decision Trees (Regressor and Classifier)
* Gradient Boosted Trees (Regressor and Classifier)
* Random Forest (Regressor and Classifier)
* Linear Regression
* Logistic Regression
* Multi-Layer Perceptron Classifier
* Support Vector Machines
* XGBoost (Regressor and Classifier)

> NOTE: With the upgrade to Spark 3 (Scala 2.12) LightGBM is **no longer supported** but will be added in a future release. 

## Documentation

Scala API documentation can be found [here](APIDOCS.md)

Python API documentation can be found [here](python/docs/APIDOCs.md) 

Analytics Package API documentation can be found [here](ANALYSIS_TOOLS_DOCS.md)

## Installing - Recommended!
Darabricks Labs AutoML can be pulled from maven central with the following coordinates.
Example - to install 0.7.2 AutoML:
```
<dependency>
  <groupId>com.databricks.labs</groupId>
  <artifactId>automl-toolkit_2.12</artifactId>
  <version>0.8.1</version>
</dependency>
```

## Building

Databricks Labs AutoML can be build with either [SBT](https://www.scala-sbt.org/) or [Maven](https://maven.apache.org/).

```text
This package requires Java 1.8.x  and scala 2.12.x to be installed on your system prior to building.
```

After cloning this repo onto your local system, navigate to the root directory and execute either:

##### Maven Build
```sbtshell
mvn clean install -DskipTests
```

##### SBT Build
```sbtshell
sbt package
```
If there is any StackOverflowError during the build, adjust the stack size on your computer's JVM. Example:
```sbtshell
#For Maven
export MAVEN_OPTS=-Xss2m
#For SBT
export SBT_OPTS="-Xss2M"
```


This will skip unit test execution (it is not recommended to run unit tests in local mode against this package as unit testing is asynchronous and incredibly CPU intensive for this code base.)


## Setup

Once the artifact has been built, attach to the Databricks Shard through either the [DBFS API](https://docs.databricks.com/api/latest/dbfs.html) or the GUI.  Once loaded into the account, utilize either the [Libraries API](https://docs.databricks.com/api/latest/libraries.html#install) to attach to a cluster, or utilize the GUI to attach the .jar to the cluster.

```text
NOTE: It is not recommended to attach this libarary to all clusters on the account.  

Use of an ML Runtime cluster configuration is highly advised to ensure that custom management of dependent 
libraries and configurations are provided 'out of the box'

```

Attach the following libraries to the cluster:
* The automl toolkit jar created above. (automatedml_2.12-((version)).jar)
* If using the PySpark API for the toolkit, the [.whl file](python/docs/APIDOCs.md#Setup) for the PySpark API.

> IMPORTANT NOTE: as of release 0.7.1, the mlflow libraries in pypi and Maven are NO LONGER NEEDED.  Attaching them
> to your cluster WILL prevent the run from logging and will throw an exception.  DO NOT ATTACH EITHER OF THEM.

## Getting Started

This package provides a number of different levels of API interaction, from the highest-level "default only" FamilyRunner to low-level APIs that allow for highly customizable workflows to be created for automated ML tuning and Inference.

Since v0.6.0 we have included an API to work with the pipeline semantics around feature engineering steps and full predict pipelines.For the purposes of a quick-start intro, the below example is of the highest-level API access point.

```scala
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.executor.FamilyRunner
import org.apache.spark.ml.PipelineModel

val data = spark.table("ben_demo.adult_data")
val overrides = Map(
  "labelCol" -> "label",
  "mlFlowLoggingFlag" -> false,
  "scalingFlag" -> true,
  "oneHotEncodeFlag" -> true,
  "pipelineDebugFlag" -> true
)
val randomForestConfig = ConfigurationGenerator
        .generateConfigFromMap("RandomForest", "classifier", overrides)

val runner = FamilyRunner(data, Array(randomForestConfig)).executeWithPipeline()

runner.bestPipelineModel("RandomForest").transform(data)

//Serialize it
runner.bestPipelineModel("RandomForest").write.overwrite().save("tmp/predict-pipeline-1")

// Load it for running inference
val pipelineModel = PipelineModel.load("tmp/predict-pipeline-1")
val predictDf = pipelineModel.transform(data)
```
This example will take the default configuration for all of the application parameters (excepting the overridden parameters in overrides Map) and execute Data Preparation tasks, Feature Vectorization, and automatic tuning of all 3 specified model types.  At the conclusion of each run, the results and model artifacts will be logged to the mlflow location that was specified in the configuration.

For a listing of all available parameter overrides and their functionality, see the [Developer Docs](APIDOCS.md)

### Inference via Mlflow Run ID
It is also possible to use MlFlow Run ID for inference, if Mlflow logging is turned on during training.
For usage, see [this](PIPELINE_API_DOCS.md#running-inference-pipeline-directly-against-an-mlflow-run-id-since-v061)

For all available pipeline APIs. please see [Developer Docs](PIPELINE_API_DOCS.md)

## Feedback

Issues with the application?  Found a bug?  Have a great idea for an addition?
Feel free to file an issue or contact [Ben](mailto:benjamin.wilson@databricks.com)

## Contributing
Have a great idea that you want to add?  Fork the repo and submit a PR!

## Legal Information
This software is provided as-is and is not officially supported by Databricks through customer technical support channels.
Support, questions, and feature requests can be communicated via email -> benjamin.wilson@databricks.com or through the Issues page of this repo.
Please see the [legal agreement](LICENSE.txt) and understand that issues with the use of this code will not be answered or investigated by Databricks Support.  

## Core Contribution team
* Lead Developer: Ben Wilson, Practice Leader, Databricks
* Developer: Daniel Tomes, RSA Practice Leader, Databricks
* Developer: Jas Bali, Sr. Solutions Consultant, Databricks
* Developer: Mary Grace Moesta, Customer Success Engineer, Databricks
* Developer: Nick Senno, Resident Solutions Architect, Databricks
