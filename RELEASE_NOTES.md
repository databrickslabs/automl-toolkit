## Auto ML Toolkit Release Notes

### Version 0.5.2

#### Fix XGBoost 0.9.0 issue with classifiers
XGBoost implementation for Spark has a default override for missing value imputation that is not compatible with SparkML.
Modifying the default behavior of XGBoost allows the new version to work correctly.
#### Adding support for NAFill:
##### New Modes: 
* <i>"auto"</i> - previously the only mode (uses statistical options to infer missing data) Usage of .setNumericFillStat and .setCharacterFillStat will inform the type of statistics that will be used to fill.
* <i>"mapFill"</i> - Custom by-column overrides.  Column names specified in either of the two maps (numericNAFillMap and/or categoricalNAFillMap) MUST be present in the DataFrame schema.  
    All fields not 	specified in these maps will use the stastics-based approach to fill na's.
* <i>"blanketFillAll"</i> - Fills na's throughout the DataFrame with values specified in characterNABlanketFillValue and numericNABlanketFillValue.
* <i>"blanketFillCharOnly"</i> - will use the characterNABlanketFillValue for only categorical columns, while the stats method will be used for numerics.
* <i>"blanketFillNumOnly"</i> - will use the numericNABlanketFillValue for only categorical columns, while the stats method will be used for character columns.
#### Safety Checks for High Cardinality columns
To prevent against extreme feature vector size 'explosion' with improperly defined high cardinality feature fields (i.e. a userID or telephone number is in the data set), cardinality checks are now enabled during feature vector creation.  The behavior of these is controlled with the following setters:
- <i>setCardinalityCheck</i> (default true) - disables or enables this feature
- <i>setCardinalityCheckMode</i> - either "silent" or "warn"
    * <u>Silent mode</u>: Removes high cardinality non-numeric fields from the feature vector.
    * <u>Warn mode</u>: If a non-numeric field is found that is above the specified threshold, an exception is thrown.
- <i>setCardinalityLimit</i> Integer limit above which the field will be removed (silent mode) or an exception will be thrown (warn mode)
- <i>setCardinalityPrecision</i> Optional override to the precision for checking the approx distinct cardinality nature of a non-numeric field.
- <i>setCardinalityType</i> Whether to use distinct or approx_distinct (approx_distinct is <i><b>highly recommended</b></i> for large data sets)
#### KSampling (Distributed SMOTE)
Now supported in the autoML Toolkit is 'intelligent minority class oversampling'. 
- This is a new train / test split method for classification problems with heavy class imbalance.
- To use, simply specify in the configuration map:
```scala
val overrides = Map(
"labelCol" -> "myLabel",
"tunerTrainSplitMethod" -> "kSample"
)

val myConfig = ConfigurationGenerator.generateConfigFromMap("XGBoost", "classifier", overrides)

val runner = FamilyRunner(data, Array(myConfig)).execute

```
Further details of the implementation, performance, and usage of this new feature will be extensively documented in an upcoming blog post by Databricks.