import unittest
from py_auto_ml.test.local_spark_singleton import SparkSingleton
from py_auto_ml.exploration.feature_importance import FeatureImportance


class TestFeatureImportance(unittest.TestCase):

    def setup(self):
        self.spark = SparkSingleton.get_instance()

    def test_bring_in_returns(self):
        self.setup()

        importances_data_frame = self.spark.createDataFrame([(1, 2, 3)], ["feature", "col2", "col3"])
        importances_data_frame.createOrReplaceTempView("importances")

        feat_imp = FeatureImportance()
        feat_imp.feature_importance = True
        feat_imp.bring_in_returns()

        assert len(feat_imp.importances.columns) == 3
        assert len(feat_imp.top_fields.columns) == 1

        self.tear_down()

    def tear_down(self):
        self.spark.stop()