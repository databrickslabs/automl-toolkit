package com.databricks.labs.automl.sanitize

import com.databricks.labs.automl.params.ManualFilters
import com.databricks.labs.automl.pipeline.Sample
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, AutomationUnitTestsUtil}

class OutlierFilteringTest extends AbstractUnitSpec {

  final private val EXPECTED_FILTER_COUNT = 1
  final private val EXPECTED_FILTER_COUNT_BOTH = 2
  final private val EXPECTED_PRESERVE_COUNT = 20
  final private val EXPECTED_PRESERVE_COUNT_BOTH = 19
  final private val UPPER_FILTER_COL_A_VALUE = 5000000.0
  final private val LOWER_FILTER_COL_A_VALUE = 0.0
  final private val BOTH_FILTER_COL_A_VALUE = Array(5000000.0, 0.0)
  final private val BOTH_FILTER_COL_B_VALUE = Array(50.0, 1.0)
  final private val UPPER_FILTER_COL_C_MANUAL_VALUE = 1000.99
  final private val LABEL_COL = "label"
  final private val FILTER_COL = "a"
  final private val EXCLUSION_FIELD = "b"
  final private val MANUAL_FIELD = "c"
  final private val EXCLUSION_COLS = Array("a", "c")
  final private val MANUAL_FILTERS = List(ManualFilters(MANUAL_FIELD, 900.0))

  private def generateOutlierData = {

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    Seq(
      Sample(0.0, 9.0, 0.99, 2, 1L),
      Sample(1.0, 8.0, 10.99, 2, 1L),
      Sample(2.0, 7.0, 0.99, 2, 1L),
      Sample(3.0, 6.0, 10.99, 2, 1L),
      Sample(4.0, 5.0, 0.99, 3, 1L),
      Sample(5.0, 4.0, 10.99, 3, 1L),
      Sample(6.0, 3.0, 10.99, 3, 1L),
      Sample(10.0, 2.0, 20.99, 3, 1L),
      Sample(20.0, 1.0, 20.99, 4, 1L),
      Sample(30.0, 2.0, 20.99, 5, 1L),
      Sample(40.0, 3.0, 20.99, 4, 1L),
      Sample(50.0, 4.0, 40.99, 4, 1L),
      Sample(60.0, 5.0, 40.99, 5, 1L),
      Sample(100.0, 6.0, 30.99, 1, 2L),
      Sample(200.0, 7.0, 30.99, 1, 3L),
      Sample(300.0, 8.0, 20.99, 1, 4L),
      Sample(1000.0, 9.0, 10.99, 3, 5L),
      Sample(10000.0, 10.0, 10.99, 4, 6L),
      Sample(100000.0, 20.0, 10.99, 3, 7L),
      Sample(1000000.0, 25.0, 1000.99, 10000, 8L),
      Sample(5000000.0, 50.0, 1.0, 17, 10L)
    ).toDF()

  }

  it should "filter appropriate values from an exponential distribution in 'upper' mode at 95p" in {

    val exponentialData = generateOutlierData

    exponentialData.show(20)

    val outlierTransformer = new OutlierFiltering(exponentialData)
      .setLabelCol(LABEL_COL)
      .setFilterBounds("upper")
      .setUpperFilterNTile(0.95)
      .setLowerFilterNTile(0.4)
      .setFilterPrecision(0.01)
      .setContinuousDataThreshold(5)
      .setParallelism(1)

    val filteredHigh = outlierTransformer.filterContinuousOutliers(
      Array(LABEL_COL, AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL),
      Array.empty[String]
    )

    val filterCount = filteredHigh._2.count()
    val nonFilterCount = filteredHigh._1.count()
    val filteredColA =
      filteredHigh._2.collect().map(_.getAs[Double](FILTER_COL)).head

    assert(
      nonFilterCount == EXPECTED_PRESERVE_COUNT,
      s"rows of non-filtered outlier data in the 95p upper mode."
    )
    assert(
      filterCount == EXPECTED_FILTER_COUNT,
      s"rows of outlier filtered data in the 95p upper mode."
    )
    assert(
      filteredColA == UPPER_FILTER_COL_A_VALUE,
      s"for the correct value of col $FILTER_COL row to be filtered out in the 95p upper mode."
    )
  }

  it should "filter appropriate values from an exponential distribution in 'lower' mode at 5p" in {

    val exponentialData = generateOutlierData

    exponentialData.show(20)

    val outlierTransformer = new OutlierFiltering(exponentialData)
      .setLabelCol("label")
      .setFilterBounds("lower")
      .setUpperFilterNTile(0.95)
      .setLowerFilterNTile(0.05)
      .setFilterPrecision(0.01)
      .setContinuousDataThreshold(5)
      .setParallelism(1)

    val filteredHigh = outlierTransformer.filterContinuousOutliers(
      Array("label", AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL),
      Array.empty[String]
    )

    val filterCount = filteredHigh._2.count()
    val nonFilterCount = filteredHigh._1.count()
    val filteredColA =
      filteredHigh._2.collect().map(_.getAs[Double](FILTER_COL)).head

    assert(
      nonFilterCount == EXPECTED_PRESERVE_COUNT,
      s"rows of non-filtered outlier data in the 95p upper mode."
    )
    assert(
      filterCount == EXPECTED_FILTER_COUNT,
      s"rows of outlier filtered data in the 95p upper mode."
    )
    assert(
      filteredColA == LOWER_FILTER_COL_A_VALUE,
      s"for the correct value of col $FILTER_COL row to be filtered out in the 5p lower mode."
    )
  }

  it should "filter appropriate values from an exponential distribution in 'both' mode at 5p and 95p" in {

    val exponentialData = generateOutlierData

    exponentialData.show(20)

    val outlierTransformer = new OutlierFiltering(exponentialData)
      .setLabelCol("label")
      .setFilterBounds("both")
      .setUpperFilterNTile(0.95)
      .setLowerFilterNTile(0.05)
      .setFilterPrecision(0.01)
      .setContinuousDataThreshold(5)
      .setParallelism(1)

    val filteredHigh = outlierTransformer.filterContinuousOutliers(
      Array("label", AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL),
      Array.empty[String]
    )

    val filterCount = filteredHigh._2.count()
    val nonFilterCount = filteredHigh._1.count()
    val filteredColA =
      filteredHigh._2.collect().map(_.getAs[Double](FILTER_COL))

    assert(
      nonFilterCount == EXPECTED_PRESERVE_COUNT_BOTH,
      s"rows of non-filtered outlier data in the 95p upper mode."
    )
    assert(
      filterCount == EXPECTED_FILTER_COUNT_BOTH,
      s"rows of outlier filtered data in the 95p upper mode."
    )
    assert(
      filteredColA.sameElements(BOTH_FILTER_COL_A_VALUE),
      s"for the correct value of col $FILTER_COL row to be filtered out in the 5p/95p both mode."
    )
  }

  it should "filter appropriate values with column exclusion in 'both' mode." in {

    val exponentialData = generateOutlierData

    exponentialData.show(20)

    val outlierTransformer = new OutlierFiltering(exponentialData)
      .setLabelCol("label")
      .setFilterBounds("both")
      .setUpperFilterNTile(0.95)
      .setLowerFilterNTile(0.05)
      .setFilterPrecision(0.01)
      .setContinuousDataThreshold(5)
      .setParallelism(1)

    val filteredHigh = outlierTransformer.filterContinuousOutliers(
      Array("label", AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL),
      EXCLUSION_COLS
    )

    val filterCount = filteredHigh._2.count()
    val nonFilterCount = filteredHigh._1.count()
    val filteredColA =
      filteredHigh._2.collect().map(_.getAs[Double](EXCLUSION_FIELD))

    filteredHigh._2.show()

    assert(
      nonFilterCount == EXPECTED_PRESERVE_COUNT_BOTH,
      s"rows of non-filtered outlier data in the 95p upper mode."
    )
    assert(
      filterCount == EXPECTED_FILTER_COUNT_BOTH,
      s"rows of outlier filtered data in the 95p upper mode."
    )
    assert(
      filteredColA.sameElements(BOTH_FILTER_COL_B_VALUE),
      s"for the correct value of col $EXCLUSION_FIELD row to be filtered out in the 95p upper mode."
    )
  }

  it should "filter appropriate values manually in 'upper' mode." in {

    val exponentialData = generateOutlierData

    exponentialData.show(20)

    val outlierTransformer = new OutlierFiltering(exponentialData)
      .setLabelCol("label")
      .setFilterBounds("upper")
      .setUpperFilterNTile(0.95)
      .setLowerFilterNTile(0.05)
      .setFilterPrecision(0.01)
      .setContinuousDataThreshold(5)
      .setParallelism(1)

    val filteredHigh = outlierTransformer.filterContinuousOutliers(
      MANUAL_FILTERS,
      Array("label", AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    )

    val filterCount = filteredHigh._2.count()
    val nonFilterCount = filteredHigh._1.count()
    val filteredData =
      filteredHigh._2.collect().map(_.getAs[Double](MANUAL_FIELD)).head

    filteredHigh._2.show()

    assert(
      nonFilterCount == EXPECTED_PRESERVE_COUNT,
      s"rows of non-filtered outlier data in the 95p upper mode."
    )
    assert(
      filterCount == EXPECTED_FILTER_COUNT,
      s"rows of outlier filtered data in the 95p upper mode."
    )
    assert(
      filteredData == UPPER_FILTER_COL_C_MANUAL_VALUE,
      s"for the correct value of col $MANUAL_FIELD row to be filtered out in manual mode."
    )
  }

}
