package com.databricks.spark.automatedml.inference

import com.databricks.spark.automatedml.executor.AutomationConfig
import com.databricks.spark.automatedml.sanitize.{DataSanitizer, VarianceFiltering}
import com.databricks.spark.automatedml.utils.{AutomationTools, DataValidation}
import org.apache.spark.sql.DataFrame

class InferencePipeline(df: DataFrame) extends AutomationConfig with AutomationTools with DataValidation{


  /**
    *
    * Replayability
    *
    * 1. Field Casting
    *     1. Need a Map of all fields and what they should be converted to
    *     2. Call base method for doing this conversion.
    * 2. NA Fill
    *     1. Map of character cols and what they should be mapped to
    *     2. Map of numeric cols and what they should be mapped to
    * 3. Variance Filter
    *     1. Array of fields to remove
    *     2. Method for removing those fields from the input Dataframe
    * 4. Outlier Filtering
    *     1. Map of thresholds to filter against
    *         1. ColumnName -> (filterValue, direction)
    *         2. Method for removing the rows that are outside of those thresholds.
    * 5. Create Feature Vector
    * 6. Covariance Filtering
    *     1. Array of columns to remove
    *     2. Re-use column filtering from variance filter
    *     3. Re-create Feature Vector
    * 7. Pearson Filtering
    *     1. Array of columns to remove
    *     2. Reuse column filtering
    *     3. Re-create Feature Vector
    * 8. OneHotEncode
    *     1. Re-create Feature Vector
    * 9. Scaling
    *     1. Re-create Feature Vector
    * 10. Model load
    *     1. need RunID as logged by MLFlow
    *     2. Retrieve model artifact
    *     3. Load as appropriate type
    *     4. Predict on main DataFrame
    *     5. Save Results
    *     6. Exit
    */










}
