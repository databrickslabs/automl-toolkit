from pyspark.sql import SparkSession


class SparkSingleton:

    @classmethod
    def get_instance(cls):
        return SparkSession.builder.getOrCreate()
