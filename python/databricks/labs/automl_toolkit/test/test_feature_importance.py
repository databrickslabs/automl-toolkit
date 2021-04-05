import unittest
from databricks.labs.automl_toolkit.test.local_spark_singleton import SparkSingleton
from databricks.labs.automl_toolkit.exploration.feature_importance import FeatureImportance
from pyspark.sql import SparkSession


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

    @staticmethod
    def convert_csv_to_df(csv_path: str):
        spark_session = SparkSession.builder.master('local[*]').appName("providentiaml-unit-tests").getOrCreate()
        spark_session.sparkContext.setLogLevel("ERROR")
        return spark_session.read.format('csv').option("header", "true").option("inferSchema", "true").load(csv_path)

    def test_loan_risk_xgboost(self):
        self.setup()
        loan_risk_df = self.convert_csv_to_df("Desktop/providenc/load_risk.csv")
        generic_overrides = {
          "labelCol": "label",
          "scoringMetric": "areaUnderROC",
          "dataPrepCachingFlag": False,
          "autoStoppingFlag": True,
          "tunerAutoStoppingScore": 0.91,
          "tunerParallelism": 1*2,
          "tunerKFold": 2,
          "tunerSeed": 42,
          "tunerInitialGenerationArraySeed": 42,
          "tunerTrainPortion": 0.7,
          "tunerTrainSplitMethod": "stratified",
          "tunerInitialGenerationMode": "permutations",
          "tunerInitialGenerationPermutationCount": 8,
          "tunerInitialGenerationIndexMixingMode": "linear",
          "tunerFirstGenerationGenePool": 16,
          "tunerNumberOfGenerations": 3,
          "tunerNumberOfParentsToRetain": 2,
          "tunerNumberOfMutationsPerGeneration": 4,
          "tunerGeneticMixing": 0.8,
          "tunerGenerationalMutationStrategy": "fixed",
          "tunerEvolutionStrategy": "batch",
          "tunerHyperSpaceInferenceFlag": True,
          "tunerHyperSpaceInferenceCount": 400000,
          "tunerHyperSpaceModelType": "XGBoost",
          "tunerHyperSpaceModelCount": 8,
          "mlFlowLoggingFlag": False,
          "mlFlowLogArtifactsFlag": False
          }
        feat_imp = FeatureImportance.run_feature_importance("XGBoost", "classifier", loan_risk_df, 20.0, "count", generic_overrides)

        assert len(feat_imp.top_fields.columns) != 0
        assert len(feat_imp.importances.columns) !=0



        self.tear_down()