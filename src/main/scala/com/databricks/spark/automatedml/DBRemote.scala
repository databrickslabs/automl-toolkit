package com.databricks.spark.automatedml

import org.apache.spark.sql.SparkSession

object DBRemote {

    def main(args: Array[String]): Unit = {
      val spark = SparkSession.builder()
        .master("local")
        .config("spark.databricks.service.client.enabled", "true")
        .config("spark.databricks.service.address", "https://demo.cloud.databricks.com")
        .config("spark.databricks.service.token", "dapi229982c386a534c9bf73581ec87da02f")
        .config("spark.databricks.service.clusterId", "1102-184351-such527")
        .config("spark.databricks.service.orgId", "83127xxxxxxxx") // Only necessary on consolidated and Azure
        //.config("spark.databricks.service.port", "8787") // Only do this on Azure
        .getOrCreate();
      println(spark.range(100).count())  // The Spark code will execute on the Databricks cluster.
    }


}


