import json
from pyspark.sql.functions import DataFrame
from python.py_auto_ml.local_spark_singleton import SparkSingleton


class FeatureImportance:
    def __init__(self,
                 # model_family: str,
                 # prediction_type: str,
                 # df: DataFrame,
                 # cutoff_value: float,
                 # cutoff_type: str,
                 # overrides=None
                 ):
        self.spark = SparkSingleton.get_instance()
        # Run feature importances
        # self.run_feature_importance(model_family,
        #                             prediction_type,
        #                             df,
        #                             cutoff_value,
        #                             cutoff_type,
        #                             overrides)
        # Get Returns as attributes of the class
        # self.bring_in_returns()

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
        self._get_returns()

    def _get_returns(self):
        """

        :return: importances dataframe with top X fields and feature importance measure based on algorithm
        :return: top_fields dataframe with the top X fields based on feature importance algorithm
        """
        if self.feature_importance != True:
            raise Exception ("Please first generate feature importances by running `run_feature_importance`")
        else:
            self.importances = self.spark.sql("select * from importances")
            self.top_fields = self.spark.sql("select feature from importances")


