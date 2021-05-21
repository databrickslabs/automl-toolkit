# Tree Model Visualization Tool

This toolkit provides a range of functionality surrounding SparkML Tree-based models:

* Extracting all available metrics from a tree or a forest of trees
* Extracting the Tree-structure, renaming the vector index positions to original qualified field names, and returning the structure and data.
* Retrieving feature importances
* Parsing a PipelineModel and extractin the tree-based model automatically.
* Calculating Shapley Values for pre-built Models and Modeling Pipelines in SparkML

## API description

### Constructor - Model-based API
Initialization of the tool is through the following signature:
```scala
TreeModelVisualization(
    model: [T],
    mode: [String],
    vectorAssembler: <Option>[VectorAssembler],
    vectorInputCols: <Option>[Array[String]]
  )
```

### Constructor - Pipeline-based API
```scala
TreePipelineVisualization(
    pipeline: PipelineModel, 
    mode: String
  )
```

The model type [T] refers to the fact that it can accept either:
* A tree-based model (DecisionTreeClassifier, RandomForestRegressor, etc...) 
OR
* A pre-fit PipelineModel that contains a tree-based model.

The 'mode' parameter dictates the type of SVG JS visualization that will be returned when using those methods  
    - (either "static" (for small trees) or "dynamic"(for larger / deeper trees) or "lightweight" (for a hybrid approach))

The visualization API for tree-based SparkML models has the following methods available to it:
1. extractAllTreeDataAsString -> a 'smart parser' that will return the decision tree as an if/else statement block. 
(effectively .toDebugString, but with column names applied instead of index positions within the vector)
2. extractAllTreeVisualization -> will automatically extract the tree data and metrics associated with each node / leaf split and apply proper column names from the vector positions, 
returning the data in a case class construct as well as collected visualization JavaScript to use through a displayHTML() command.
This is an Array[VisualizationOutput] type and as such, will have to have its elements extracted individually to display tree data for forest algorithms.
3. extractFirstTreeVisualization -> extracts the first tree (the only tree for a decision tree model) for the decision data as well as the html JS code for displaying.
4. extractImportancesAsTable -> generates an html table that will show the ranked list of feature importances by initial column names.
5. extractImportancesAsChart -> generates a d3.js chart of the feature importances to use in a displayHTML() command.

## Demo notebook with API examples
[Demo in .dbc format](demo/MLVisualizations.dbc) | 
[Demo in .html format](demo/MLVisualizations.html)
```text
note: The .html format will not render the visualizations correctly if directly opened.  You must attach and execute
the imported notebook in order to see the tree graphs from within Databricks.
```


## Example:

```scala

import org.apache.spark.ml.regression.{DecisionTreeRegressor, RandomForestRegressor, GBTRegressor}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, RandomForestRegressionModel, GBTRegressionModel}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.DataFrame

/**
 * Function for detecting categorical strings and generating indexers dynamically
 */
def applyIndexers(data: DataFrame): Array[StringIndexer] = {
  data.schema.collect{ case x if x.dataType == StringType => x.name}.map(x => new StringIndexer().setInputCol(x).setOutputCol(s"${x}_si").setHandleInvalid("keep")).toArray
}

/**
 * Function for building the Vector Assembler based on 'seen' indexers and feature fields
 */
def buildVectorAssembler(initialFields: Array[String], indexers: Array[StringIndexer], outputCol: String): VectorAssembler = {
  val updatedFields = initialFields.filterNot(x => indexers.map(_.getInputCol).contains(x)) ++ indexers.map(_.getOutputCol)
  new VectorAssembler().setInputCols(updatedFields).setOutputCol(outputCol)
}

val data = spark.table("BenWDatabase.ml_abalone")
val LABEL_COL = "age"
val FEATURES_COL = "features"
val initialFeatures = data.schema.names.filterNot(_ == LABEL_COL)

val indexers = applyIndexers(data)

val indexerPipeline = new Pipeline().setStages(indexers)

val preStagePipeline = new Pipeline().setStages(Array(indexerPipeline) ++ Array(buildVectorAssembler(initialFeatures, indexers.toArray, "features")))

val dtModel = new DecisionTreeRegressor()
  .setFeaturesCol(FEATURES_COL)
  .setLabelCol(LABEL_COL)
  .setPredictionCol("dtPred")
  .setImpurity("variance")
  .setMaxBins(20)
  .setMaxDepth(4)
  .setMinInfoGain(1e-7)
  .setMinInstancesPerNode(5)
  .setSeed(42L)

val rfModel = new RandomForestRegressor()
  .setFeatureSubsetStrategy("onethird")
  .setFeaturesCol(FEATURES_COL)
  .setLabelCol(LABEL_COL)
  .setPredictionCol("rfPred")
  .setImpurity("variance")
  .setMaxBins(20)
  .setMaxDepth(4)
  .setMinInfoGain(1e-7)
  .setMinInstancesPerNode(5)
  .setNumTrees(10)
  .setSeed(42L)
  .setSubsamplingRate(0.8)

val gbtModel = new GBTRegressor()
  .setLabelCol(LABEL_COL)
  .setFeaturesCol(FEATURES_COL)
  .setImpurity("gini")
  .setLossType("squared")
  .setMaxBins(20)
  .setMaxDepth(4)
  .setMaxIter(100)
  .setMinInfoGain(1e-7)
  .setMinInstancesPerNode(5)
  .setStepSize(1e-3)

val dtModelingPipeline = new Pipeline().setStages(Array(preStagePipeline, dtModel))
val rfModelingPipeline = new Pipeline().setStages(Array(preStagePipeline, rfModel))
val gbtModelingPipeline = new Pipeline().setStages(Array(preStagePipeline, gbtModel))

val dtFit = dtModelingPipeline.fit(data)
val rfFit = rfModelingPipeline.fit(data)
val gbtFit = gbtModelingPipeline.fit(data)

val dtBuiltModel = dtFit.stages.last.asInstanceOf[DecisionTreeRegressionModel]
val rfBuiltModel = rfFit.stages.last.asInstanceOf[RandomForestRegressionModel]
val gbtBuiltModel = gbtFit.stages.last.asInstanceOf[GBTRegressionModel]


val features = dtModelingPipeline.getStages.head.asInstanceOf[Pipeline].getStages.last.asInstanceOf[VectorAssembler].getInputCols


// -----------------------//

import com.databricks.labs.automl.exploration.analysis.trees.TreeModelVisualization

// Manual API for model + VectorAssembler
val dtVectorAssembler = dtModelingPipeline.getStages.head.asInstanceOf[Pipeline].getStages.last.asInstanceOf[VectorAssembler]
val allTreeExtract = new TreeModelVisualization(dtBuiltModel, "static", dtVectorAssembler).extractAllTreeVisualization
displayHTML(allTreeExtract(0).html)

// -----------------------// Feature Importances as Table
// Manual API for model + column listing for feature vector
val inputFields = dtVectorAssembler.getInputCols
displayHTML(TreeModelVisualization(dtBuiltModel, "dynamic", vectorInputCols=inputFields).extractImportancesAsTable)
// -----------------------//
// Pipeline API for tree Data Feature Importances
import com.databricks.labs.automl.exploration.analysis.trees.TreePipelineVisualization
displayHTML(TreePipelineVisualization(dtFit, "static").extractImportancesAsTable)

// -----------------------//
// Pipeline API for displaying all trees in forest (example showing the 60th tree in the forest)
val treesInTheForest = TreePipelineVisualization(gbtFit, "static").extractAllTreeVisualization
displayHTML(treesInTheForest(60).html)

// -----------------------//
// Pipeline API for tree Data Feature Importances as Chart
displayHTML(TreePipelineVisualization(rfFit, "dynamic").extractImportancesAsChart)

// -----------------------//
// Pipeline API for tree Data Tree Visualization - static verbose display
displayHTML(TreePipelineVisualization(rfFit, "static").extractFirstTreeVisualization)

// -----------------------//
// Pipeline API for tree Data Tree Visualization - dynamic display
displayHTML(TreePipelineVisualization(gbtFit, "static").extractFirstTreeVisualization)

```

The visualizations are supported in Databricks Notebooks and will render as interactive SVG elements.

### Distributed Approximate Shapley Values
This API is accessible through a Model-based API and a Pipeline-based API.

#### Model based API Signature
The following calculates an array of Shapley Values for each record with the first Shapley value corresponding to the 
first feature etc. An array containing the approximate error for each Shapley value is also returned
```scala
import com.databricks.labs.automl.exploration.analysis.shap.ShapleyModel

val shapPreDataModel = model.fit(data).transform(data)
val shapValuesLR = ShapleyModel(shapPreDataModel, model, featureCol, 200, 60, 11L).calculate
display(shapValuesLR)
```

The object constructor for ShapleyModel signature is as follows:
- The DataFrame that was used during model training (with feature vector)
- The fit model under test
- the name of the features column (vector)
- the repartition count (higher numbers equate to more concurrent SHAP calculations per partition)
- the number of partition-level shap calculations to perform (higher values increase accuracy at the expense of runtime)
- seed value for reproducability for candidate row selection within partitions

NOTE:
The vectorizedData DataFrame MUST include the featureCol column as a org.apache.spark.ml.feature.VectorAssembler Vector field.
The model can be one of:
- RandomForestRegressionModel
- RandomForestClassificationModel
- DecisionTreeRegressionModel
- DecisionTreeClassificationModel
- GBTRegressionModel
- GBTClassifcationModel
- LogisticRegressionModel
- LinearRegressionModel

(Other SparkML models will be supported in the future)

The featureCol is simply the name of the Vectorized feature field in the DataFrame.

Vector mutations represents the maximum number of rows to use <i>within each partition</i> for calculating the shapley values.
The random seed is a start value for the random selection of both vectors to mutate and the index shuffling algorithm for feature inclusion in mutation.

To get aggregated Shapley values for the entire dataset

```scala
val shapPreDataModel = preStagePipeline.fit(data).transform(data)
val shapValuesRF = ShapleyModel(shapPreDataModel, rfBuiltModel, FEATURES_COL, 2, 1000, 1620).getShapValuesFromModel(initialFeatures)
display(shapValuesRF)
```

Where `initialFeatures` is an `Seq[String]` of the original feature names


For further information on how calculating shapley values occurs, please see: [Interpreting ML predictions white paper](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)

#### Pipeline-based API
```scala
import com.databricks.labs.automl.exploration.analysis.shap.ShapleyPipeline

val shapPipelineData = preStagePipeline.fit(data).transform(data)
val shapValuesGBT = ShapleyPipeline(shapPipelineData, gbtFit, 400, 200, 11L).getShapValuesFromPipeline
display(shapValuesGBT)
```

The signature of the ShapleyPipeline API is:

- The DataFrame that was used during model training (with feature vector)
- The fit pipeline (containing the model to test) 
- the repartition count (higher numbers equate to more concurrent SHAP calculations per partition)
- the number of partition-level shap calculations to perform (higher values increase accuracy at the expense of runtime)
- seed value for reproducibility for candidate row selection within partitions


#### Questions / Comments / Contributions

Suggestions, feedback, and comments are welcome.  
Contributions are even more welcome.

Contact authors at benjamin.wilson@databricks.com or nick.senno@databricks.com