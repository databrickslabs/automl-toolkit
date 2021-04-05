import unittest
from databricks.labs.automl_toolkit.test.local_spark_singleton import SparkSingleton
from databricks.labs.automl_toolkit.automation_runner import AutomationRunner


class TestFamilyRunner(unittest.TestCase):

    def setup(self):
        self.spark = SparkSingleton.get_instance()

    def test_get_returns(self):
        self.setup()
        automation_runner = AutomationRunner()

        model_report_data_frame = self.spark.createDataFrame([(1,2,3)],["col1", "col2", "col3"])
        model_report_data_frame.createOrReplaceTempView("modelReport")
        model_report_data_frame.createOrReplaceTempView("modelReportData")

        generation_report_data_frame = self.spark.createDataFrame([(4, 5, 6, 7)], ["col1", "col2", "col3", "col4"])
        generation_report_data_frame.createOrReplaceTempView("generationReport")

        confusion_data = self.spark.createDataFrame([(7, 8)], ["col1", "col2"])
        confusion_data.createOrReplaceTempView("confusionData")

        prediction_data = self.spark.createDataFrame([(9,10,11,12,13)], ["col1", "col2", "col3", "col4", "col5"])
        prediction_data.createOrReplaceTempView("predictionData")

        data_with_preds = self.spark.createDataFrame([(14, 15)], ["col1", "col2"])
        data_with_preds.createOrReplaceTempView("dataWithPredictions")

        # Test with RUN
        automation_runner._automation_runner = True
        automation_runner.get_returns("run")
        assert len(automation_runner.generation_report.columns) == 4
        assert len(automation_runner.model_report.columns) == 3

        # Test with CONFUSION
        automation_runner.get_returns("confusion")
        assert len(automation_runner.confusion_data.columns) == 2
        assert len(automation_runner.prediction_data.columns) == 5
        assert len(automation_runner.generation_report.columns) == 4
        assert len(automation_runner.model_report.columns) == 3

        # Test with PREDICTION
        automation_runner.get_returns("prediction")
        assert len(automation_runner.generation_report.columns) == 4
        assert len(automation_runner.model_report.columns) == 3
        assert len(automation_runner.data_with_predictions.columns) == 2

        self.tear_down()

    def tear_down(self):
        self.spark.stop()
