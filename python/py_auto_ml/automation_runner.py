import json
from pyspark.sql.functions import DataFrame
from py_auto_ml.spark_singleton import SparkSingleton


class AutomationRunner:

    def __init__(self,
                 model_family: str,
                 prediction_type: str,
                 df: DataFrame,
                 runner_type: str,
                 overrides=None):
        # Setup Spark singleton Instance
        self.spark = SparkSingleton.get_instance()
        # Run automation runner
        self.run_automation_runner(model_family,
                                   prediction_type,
                                   df,
                                   runner_type,
                                   overrides)
        # Bring in the returns attributes of the class
        self._bring_in_returns(runner_type.lower())


    def run_automation_runner(self, model_family: str,
                              prediction_type: str,
                              df: DataFrame,
                              runner_type: str,
                              overrides=None):

        runner_type_lower = runner_type.lower()
        self._check_runner_types(runner_type_lower)

        # Check if you need default instance config or generating from map of overrides
        if overrides is not None:
            default_flag = "false"
            # Stringify overrides to JSON
            stringified_overrides = json.dumps(overrides)
        else:
            default_flag = "true"
            stringified_overrides = ""

        self.spark._jvm.com.databricks.labs.automl.pyspark.AutomationRunnerUtil.runAutomationRunner(model_family,
                                                                                                        prediction_type,
                                                                                                        stringified_overrides,
                                                                                                        df._jdf,
                                                                                                        runner_type_lower,
                                                                                                        default_flag)

    def _bring_in_returns(self,
                          runner_type: str):
        # Cache the returns
        if runner_type == "run":
            self.generation_report = self.spark.sql("select * from generationReport")
            self.model_report = self.spark.sql("select * from modelReport")
        elif runner_type == "confusion":
            self.confusion_data = self.spark.sql("select * from confusionData")
            self.prediction_data = self.spark.sql("select * from predictionData")
            self.generation_report = self.spark.sql("select * from generationReport")
            self.model_report = self.spark.sql("select * from modelReport")
        elif runner_type == "prediction":
            self.data_with_predictions = self.spark.sql("select * from dataWithPredictions")
            self.generation_report = self.spark.sql("select * from generationReport")
            self.model_report = self.spark.sql("select * from modelReportData")
        else:
            print("No returns were added - check your runner_type")

    @staticmethod
    def _check_runner_types(runner_type: str):
        acceptable_strings = ["run", "confusion", "prediction"]
        if runner_type not in acceptable_strings:
            raise Exception("runner_type must be one of the following run, confusion, or prediction")
        else:
            None