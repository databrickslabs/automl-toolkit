import unittest
from databricks.labs.automl_toolkit.test.local_spark_singleton import SparkSingleton
from databricks.labs.automl_toolkit.executor.family_runner import FamilyRunner


class TestFamilyRunner(unittest.TestCase):

    def setup(self):
        self.spark = SparkSingleton.get_instance()

    def test_get_returns(self):
        self.setup()
        family_runner = FamilyRunner()

        model_report_data_frame = self.spark.createDataFrame([(1,2,3)],["col1", "col2", "col3"])
        model_report_data_frame.createOrReplaceTempView("modelReportDataFrame")

        generation_report_data_frame = self.spark.createDataFrame([(4, 5, 6, 7)], ["col1", "col2", "col3", "col4"])
        generation_report_data_frame.createOrReplaceTempView("generationReportDataFrame")

        best_mlflow_run_id = self.spark.createDataFrame([(7, 8)], ["col1", "col2"])
        best_mlflow_run_id.createOrReplaceTempView("bestMlFlowRunId")

        family_runner._family_runner = True
        family_runner.get_returns()

        assert len(family_runner.model_report.columns) == 3
        assert len(family_runner.best_mlflow_run_id.columns) == 2
        assert len(family_runner.generation_report.columns) == 4

        self.tear_down()

    def tear_down(self):
        self.spark.stop()
