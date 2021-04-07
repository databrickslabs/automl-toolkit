import json
from pyspark.sql.functions import DataFrame
from databricks.labs.automl_toolkit.local_spark_singleton import SparkSingleton
from databricks.labs.automl_toolkit.utilities.helpers import Helpers


class FeatureImportance:
    def __init__(self):
        self.spark = SparkSingleton.get_instance()


    def run_feature_importance(self,
                               model_family: str,
                               prediction_type: str,
                               dataframe: DataFrame,
                               cutoff_value: float,
                               cutoff_type: str,
                               overrides=None):
        """

        :param model_family: str
            One of the supported model types

        :param prediction_type: str
            Either "classifier" or "regressor"

        :param df: DataFrame

        :param cutoff_value: float
            Threshold value for feature importance algorithm

        :param cutoff_type: str
            Cutoff for the number features

        :param overrides: dict
            Dictionary of configuration overrides

        :return:
        """

        # Checking for supported model families and types
        Helpers.check_model_family(model_family)
        Helpers.check_prediction_type(prediction_type)


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
                                                                                                      dataframe._jdf,
                                                                                                      cutoff_type,
                                                                                                      cutoff_value,
                                                                                                      default_flag)
        self.feature_importance = True
        return self._get_returns()

    def _get_returns(self):
        """

        :return: dict of dataframes:
            'importances': importances df
            'top_fields': top fields df
        """
        if self.feature_importance != True:
            raise Exception ("Please first generate feature importances by running `run_feature_importance`")
        else:
            importances = self.spark.sql("select * from importances")
            top_fields = self.spark.sql("select feature from importances")
            return_dict = {
                'importances': importances,
                'top_fields': top_fields
            }
            return return_dict


