from databricks.labs.automl_toolkit.local_spark_singleton import SparkSingleton
from pyspark.ml.common import _java2py


class Shapley:
    def __init__(self,
                 feature_data,
                 model,
                 feature_col,
                 repartition_value,
                 vector_mutations,
                 random_seed=1621
                 ):

        self._spark = SparkSingleton.get_instance()
        self._sc = self._spark.sparkContext

        shapley_model = self._spark._jvm.com.databricks.labs.automl.exploration.analysis.shap.ShapleyModel
        self._shapley_model = shapley_model(feature_data._jdf,
                                            model._java_obj,
                                            feature_col,
                                            repartition_value,
                                            vector_mutations,
                                            random_seed
                                            )

    def calculate(self):

        return _java2py(self._sc, self._shapley_model.calculate())

    def feature_aggregated_shap(self, input_cols):

        return _java2py(self._sc, self._shapley_model.getShapValuesFromModel(input_cols))

    def count_efficiency(self, feature_shap_data, tol=1e-3):

        return self._shapley_model.countEfficiency(feature_shap_data._jdf, tol)



