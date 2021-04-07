import json
from pyspark.sql.functions import DataFrame
from databricks.labs.automl_toolkit.local_spark_singleton import SparkSingleton
from databricks.labs.automl_toolkit.utilities.helpers import Helpers


class AutomationRunner:

    def __init__(self):
        # Setup Spark singleton Instance
        self.spark = SparkSingleton.get_instance()

    def run_automation_runner(self,
                              model_family: str,
                              prediction_type: str,
                              dataframe: DataFrame,
                              runner_type: str,
                              overrides=None):
        """

        :param model_family: str
            One of the supported model types

        :param prediction_type: str
            Either "classifier" or "regressor"

        :param dataframe: DataFrame

        :param runner_type: str
            One of the following calls to the automation runner: "run", "confusion", "prediction"

        :param overrides: dict
            Dictionary of configuration overrides

        :return:
        """
        # Checking for supported model families and types
        Helpers.check_model_family(model_family)
        Helpers.check_prediction_type(prediction_type)

        runner_type_lower = runner_type.lower()
        Helpers.check_runner_types(runner_type_lower)


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
                                                                                                    dataframe._jdf,
                                                                                                    runner_type_lower,
                                                                                                    default_flag)
        self._automation_runner = True

        return self._get_returns(runner_type_lower)

    def _get_returns(self,
                    runner_type: str):
        """

        :param runner_type:
            One of the following calls to the automation runner: "run", "confusion", "prediction"
        :return: Dataframe depending on `runner_type`
            `run`
                generation_report dataframe
                model_report dataframe
            `confusion`
                confusion_data: dataframe
                prediction_data: dataframe
                generation_report: dataframe
                model_report: dataframe
            `prediction`
                data_with_predictions: dataframe
                generation_report: dataframe
                model_report: dataframe
        """
        # Cache the returns
        if self._automation_runner == True:
            if runner_type == "run":
                generation_report = self.spark.sql("select * from generationReport")
                model_report = self.spark.sql("select * from modelReport")
                return_dict = {
                    'generation_report': generation_report,
                    "model_report": model_report
                }
                return return_dict
            elif runner_type == "confusion":
                confusion_data = self.spark.sql("select * from confusionData")
                prediction_data = self.spark.sql("select * from predictionData")
                generation_report = self.spark.sql("select * from generationReport")
                model_report = self.spark.sql("select * from modelReport")
                return_dict = {
                    'confusion_data': confusion_data,
                    'prediction_data': prediction_data,
                    'generation_report': generation_report,
                    'model_report': model_report
                }
                return return_dict
            elif runner_type == "prediction":
                data_with_predictions = self.spark.sql("select * from dataWithPredictions")
                generation_report = self.spark.sql("select * from generationReport")
                model_report = self.spark.sql("select * from modelReportData")
                return_dict = {
                    'data_with_predictions': data_with_predictions,
                    'generation_report': generation_report,
                    'model_report': model_report
                }
                return return_dict
            else:
                print("No returns were added - check your runner_type")

        else:
            raise Exception ("In order to generate the proper returns for the automation runner, please first run the "
                             "automation runner with the `run_automation_runner`")

