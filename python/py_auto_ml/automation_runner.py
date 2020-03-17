import json
from pyspark.sql.functions import DataFrame
from python.py_auto_ml.local_spark_singleton import SparkSingleton
from  python.py_auto_ml.utilities.helpers import Helpers


class AutomationRunner:

    def __init__(self,
                 # model_family: str,
                 # prediction_type: str,
                 # df: DataFrame,
                 # runner_type: str,
                 # overrides=None
                 ):
        # Setup Spark singleton Instance
        self.spark = SparkSingleton.get_instance()
        # Run automation runner
        # self._run_automation_runner(model_family,
        #                            prediction_type,
        #                            df,
        #                            runner_type,
        #                            overrides)
        # # Bring in the returns attributes of the class
        # self.get_returns(runner_type.lower())

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
                                                                                                    dataframe._jdf,
                                                                                                    runner_type_lower,
                                                                                                    default_flag)
        self._automation_runner = True

        self._get_returns(runner_type_lower)

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

        else:
            raise Exception ("In order to generate the proper returns for the automation runner, please first run the "
                             "automation runner with the `run_automation_runner`")

    @staticmethod
    def _check_runner_types(runner_type: str):
        """

        :param runner_type: str
            Checking that the runner_type is a supported runner_type
        :return:
        """
        acceptable_strings = ["run", "confusion", "prediction"]
        if runner_type not in acceptable_strings:
            raise Exception("runner_type must be one of the following run, confusion, or prediction")