import json
from pyspark.sql.functions import DataFrame
from py_auto_ml.spark_singleton import SparkSingleton


class FeatureImportance:
    def __init__(self,
                 model_family: str,
                 prediction_type: str,
                 df: DataFrame,
                 cutoff_value: float,
                 cutoff_type: str,
                 overrides=None):
        self.spark = SparkSingleton.get_instance()
        # Run feature importances
        self.run_feature_importance(model_family,
                                    prediction_type,
                                    df,
                                    cutoff_value,
                                    cutoff_type,
                                    overrides)
        # Get Returns as attributes of the class
        self._bring_in_returns()

    # class feature_importance
    def run_feature_importance(self,
                               model_family: str,
                               prediction_type: str,
                               df: DataFrame,
                               cutoff_value: float,
                               cutoff_type: str,
                               overrides=None):
        ## Set flag for default configs
        if overrides is not None:
            default_flag = "false"
            # Convert the configs to JSON
            stringified_overrides = json.dumps(overrides)
        else:
            stringified_overrides = ""
            default_flag = "true"

        # Pass to JVM to run FI
        self.spark._jvm.com.databricks.labs.automl.pyspark.FeatureImportanceUtil.runFeatureImportance(model_family,
                                                                                                      prediction_type,
                                                                                                      stringified_overrides,
                                                                                                      df._jdf,
                                                                                                      cutoff_type,
                                                                                                      cutoff_value,
                                                                                                      default_flag)

    def _bring_in_returns(self):
        self.importances = self.spark.sql("select * from importances")
        self.top_fields = self.spark.sql("select feature from importances")


