package com.databricks.labs.automl.pipeline

import java.sql.{Date, Timestamp}

import com.databricks.labs.automl.{
  AbstractUnitSpec,
  AutomationUnitTestsUtil,
  PipelineTestUtils
}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

class DateFieldTransformerTest extends AbstractUnitSpec {

  "DateFieldTransformerTest" should "should convert data/time fiels" in {
    val spark = AutomationUnitTestsUtil.sparkSession
    val dateColToBeTransformed = "download_date"
    val tsColToBeTransformed = "event_ts"
    val sourceDF = spark.createDataFrame(
      List(
        Row(
          300L,
          Date.valueOf("2016-09-30"),
          Timestamp.valueOf("2007-09-23 10:10:10.0"),
          0
        ),
        Row(
          400L,
          Date.valueOf("2016-10-30"),
          Timestamp.valueOf("2007-09-24 12:05:55.0"),
          1
        )
      ).asJava,
      StructType(
        Array(
//        StructField("download_events", ArrayType(StringType, false), nullable = true),
          StructField("download_events", LongType, nullable = true),
          StructField(dateColToBeTransformed, DateType, nullable = true),
          StructField(tsColToBeTransformed, TimestampType, nullable = true),
          StructField("label", IntegerType, nullable = false)
        )
      )
    )

    val stages = new ArrayBuffer[PipelineStage]
    stages += PipelineTestUtils
      .addZipRegisterTmpTransformerStage(
        "label",
        Array("download_events", "download_date", "event_ts")
      )
    stages += new DateFieldTransformer()
      .setLabelColumn("label")
      .setMode("split")
    val transformedDfwithDateTsFeatures = PipelineTestUtils
      .saveAndLoadPipeline(stages.toArray, sourceDF, "date-field-tran-pipeline")
      .transform(sourceDF)
    val expectedColsToBePresent =
      Array(
        "download_date_year",
        "download_date_month",
        "download_date_day",
        "event_ts_year",
        "event_ts_month",
        "event_ts_day",
        "event_ts_hour",
        "event_ts_minute",
        "event_ts_second"
      )
    assert(
      !transformedDfwithDateTsFeatures.columns.exists(
        col => Array(dateColToBeTransformed, tsColToBeTransformed).contains(col)
      ),
      s"""Original columns ${Array(dateColToBeTransformed, tsColToBeTransformed)
        .mkString(", ")} should have been dropped"""
    )
    assert(
      transformedDfwithDateTsFeatures.columns
        .exists(col => expectedColsToBePresent.contains(col)),
      s"""These columns ${dateColToBeTransformed
        .mkString(", ")} should have been added"""
    )
  }
}
