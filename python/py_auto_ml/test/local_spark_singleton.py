from pyspark.sql import SparkSession
import os


class SparkSingleton:
    """A singleton class on Datalib which returns one Spark instance"""
    __instance = None

    @classmethod
    def get_instance(cls):
        """Create a Spark instance for Datalib.
        :return: A Spark instance
        """
        return (SparkSession.builder
                .getOrCreate())

    @classmethod
    def get_local_instance(cls):
        return (SparkSession.builder
                .master("local[*]")
                .appName("automl")
                .getOrCreate())

