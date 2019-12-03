import json
from pyspark.sql.functions import DataFrame
from py_auto_ml.spark_singleton import SparkSingleton


class FamilyRunner:
    def __init__(self,
                 family_configs: dict,
                 prediction_type: str,
                 df: DataFrame):
        self.spark = SparkSingleton.get_instance()
        self.run_family_runner(family_configs,
                               prediction_type,
                               df)
        self._bring_in_returns()
        self.spark = SparkSingleton.get_instance()

    def run_family_runner(self,
                          family_configs: dict,
                          prediction_type: str,
                          df: DataFrame):
        """

        :param family_configs: dict
            Supported model_family as a key, vaue is a dictionary of configuration overrides
        :param prediction_type: str
            "regressor" or "classifier"
        :param df: dataframe
        :return:
        """
        stringified_family_configs = json.dumps(family_configs)
        self.spark._jvm.com.databricks.labs.automl.pyspark.FamilyRunnerUtil.runFamilyRunner(stringified_family_configs,
                                                                                            prediction_type,
                                                                                            df._jdf)

    def _bring_in_returns(self):
        """

        :return: model_report: dataframe
            generation_report: dataframe
            best_mlflow_run_id: dataframe
        """
        self.model_report = self.spark.sql("SELECT * FROM modelReportDataFrame")
        self.generation_report = self.spark.sql("SELECT * FROM generationReportDataFrame")
        self.best_mlflow_run_id = self.spark.sql("SELECT * FROM bestMlFlowRunId")