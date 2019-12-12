package com.databricks.labs.automl.sanitize

import com.databricks.labs.automl.params.ManualFilters
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}

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

  it should "filter appropriate values from an exponential distribution in 'upper' mode at 95p" in {

    val exponentialData = DiscreteTestDataGenerator.generateOutlierData

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

    val exponentialData = DiscreteTestDataGenerator.generateOutlierData

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

    val exponentialData = DiscreteTestDataGenerator.generateOutlierData

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

    val exponentialData = DiscreteTestDataGenerator.generateOutlierData

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

    val exponentialData = DiscreteTestDataGenerator.generateOutlierData

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
