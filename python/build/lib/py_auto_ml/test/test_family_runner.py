import unittest
from unittest import mock
from pyspark.sql.session import SparkSession
from test.local_spark_singleton import SparkSingleton
from py_auto_ml.executor.family_runner import FamilyRunner


class TestFamilyRunner(unittest.TestCase):

    def setup(self):
        self.spark = SparkSingleton.get_instance()

    def test_bring_in_returns(self):
        self.setup()
        model_report_data_frame = self.spark.createDataFrame([(1,2,3)],["col1", "col2", "col3"])
        model_report_data_frame.createOrReplaceTempView("modelReportDataFrame")

        generation_report_data_frame = self.spark.createDataFrame([(4, 5, 6)], ["col1", "col2", "col3"])
        generation_report_data_frame.createOrReplaceTempView("generationReportDataFrame")

        best_mlflow_run_id = self.spark.createDataFrame([(7, 8, 9)], ["col1", "col2", "col3"])
        best_mlflow_run_id.createOrReplaceTempView("bestMlFlowRunId")

        # print(m.model_report)

        self.assertIsNotNone(m.model_report)
        self.assertIsNotNone(m.genereation_report)
        self.assertIsNotNone(m.best_mlflow_run_id)



        self.tear_down()

    def tear_down(self):
        self.spark.stop()
