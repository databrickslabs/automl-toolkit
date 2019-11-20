import json
from pyspark.sql.functions import DataFrame
from python.spark_singleton import SparkSingleton


class family_runner:
    """
    @parameter: family_configs - a dictionary with the key being model family and value being a dictionary of overrides
    @parameter: prediction_typ - regressor or classifier
    @parameter: df - data frame to run models on
    """

    def __init__(self,
                 family_configs: dict,
                 prediction_type: str,
                 df: DataFrame):
        self.run_family_runner(family_configs,
                               prediction_type,
                               df)
        self._bring_in_returns()
        self.spark = SparkSingleton.get_instance()

    def run_family_runner(family_configs: dict,
                          prediction_type: str,
                          df: DataFrame):
        stringified_family_configs = json.dumps(family_configs)
        spark._jvm.com.databricks.labs.automl.four.pyspark(stringified_family_configs,
                                                           prediction_type,
                                                           df)

    def _bring_in_returns(self):
        self.model_report = spark.sql("SELECT * FROM modelReportDataFrame")
        self.generation_report = spark.sql("SELECT * FROM generationReportDataFrame")