package com.databricks.labs.automl

import org.apache.spark.sql.Row

class AutomationRunnerTest extends AbstractUnitSpec {

  "AutomationRunner" should "throw NullPointerException if it is instantiated with null constructor" in {
    a[NullPointerException] should be thrownBy {
      new AutomationRunner(null).runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with wrong evolution strategy" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      val fullConfig =
        AutomationUnitTestsUtil.getXgBoostConfig(adultDfwithLabel, "err")
      fullConfig.runWithConfusionReport()
    }
  }

  it should "throw AssertionError with wrong label column" in {
    a[org.apache.spark.sql.AnalysisException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setLabelCol("label_test")
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with wrong modeling family" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setModelingFamily("test")
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with wrong firstGenerationGenePool config" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setFirstGenerationGenePool(1)
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with empty input dataset with schema" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(
        AutomationUnitTestsUtil.sparkSession.createDataFrame(
          AutomationUnitTestsUtil.sparkSession.sparkContext.emptyRDD[Row],
          adultDfwithLabel.schema
        )
      ).runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with empty input dataset with no schema" in {
    a[IllegalArgumentException] should be thrownBy {
      new AutomationRunner(AutomationUnitTestsUtil.sparkSession.emptyDataFrame)
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with zero number of generations" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setNumberOfGenerations(0)
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with zero number of parents To retain" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setNumberOfParentsToRetain(0)
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with zero number of mutations per generation" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setNumberOfMutationsPerGeneration(0)
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with zero genetic mixing" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setGeneticMixing(0)
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with wrong generational mutation strategy" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setGenerationalMutationStrategy("err")
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with wrong feature importance cutoff type" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setFeatureImportanceCutoffType("err")
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with invalid continuousDiscretizerBucketCount" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setFeatureInteractionContinuousDiscretizerBucketCount(0)
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with invalid parallelism setting" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setFeatureInteractionParallelism(0)
        .runWithConfusionReport()
    }
  }

  it should "throw IllegalArgumentException with invalid retention mode setting" in {
    a[IllegalArgumentException] should be thrownBy {
      val adultDfwithLabel = AutomationUnitTestsUtil.getAdultDf()
      new AutomationRunner(adultDfwithLabel)
        .setFeatureInteractionRetentionMode("err")
        .runWithConfusionReport()
    }
  }

}
