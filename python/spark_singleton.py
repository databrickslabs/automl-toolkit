from python.sql import SparkSession
import os

class SparkSingleton:
    @classmethod
    def get_instance(cls):
        return SparkSession.builder.getOrCreate()
