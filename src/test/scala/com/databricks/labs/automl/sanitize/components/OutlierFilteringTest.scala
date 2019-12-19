package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}
import com.databricks.labs.automl.params.ManualFilters
import com.databricks.labs.automl.sanitize.OutlierFiltering
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils

class OutlierFilteringTest extends AbstractUnitSpec {

  final private val OUTLIER_ROW_COUNT = 500
  final private val OUTLIER_LABEL_DISTINCT_COUNT = 5
  final private val EXPECTED_FILTER_COUNT_UPPER = 30
  final private val EXPECTED_FILTER_COUNT_LOWER = 20
  final private val EXPECTED_FILTER_COUNT_BOTH = 50
  final private val EXPECTED_FILTER_EXCLUSION_MODE = 49
  final private val EXPECTED_FILTER_COUNT_MANUAL = 244
  final private val EXPECTED_PRESERVE_COUNT_UPPER = 470
  final private val EXPECTED_PRESERVE_COUNT_LOWER = 480
  final private val EXPECTED_PRESERVE_COUNT_BOTH = 450
  final private val EXPECTED_PRESERVE_EXCLUSION_MODE = 451
  final private val EXPECTED_PRESERVE_COUNT_MANUAL = 256
  final private val UPPER_FILTER_COL_A_VALUE = 8.30584E8
  final private val LOWER_FILTER_COL_A_VALUE = 0.0
  final private val BOTH_FILTER_COL_A_VALUE = Array(8.30584E8, 8.35896888E8,
    8.41232384E8, 8.46590536E8, 8.51971392E8, 8.57375E8, 8.62801408E8,
    8.68250664E8, 8.73722816E8, 8.79217912E8, 8.84736E8, 8.90277128E8,
    8.95841344E8, 9.01428696E8, 9.07039232E8, 9.12673E8, 9.18330048E8,
    9.24010424E8, 9.29714176E8, 9.35441352E8, 9.41192E8, 9.46966168E8,
    9.52763904E8, 9.58585256E8, 9.64430272E8, 9.70299E8, 9.76191488E8,
    9.82107784E8, 9.88047936E8, 9.94011992E8, 0.0, 8.0, 64.0, 216.0, 512.0,
    1000.0, 1728.0, 2744.0, 4096.0, 5832.0, 8000.0, 10648.0, 13824.0, 17576.0,
    21952.0, 27000.0, 32768.0, 39304.0, 46656.0, 54872.0)
  final private val BOTH_FILTER_COL_B_VALUE = Array(499.0, 498.0, 497.0, 496.0,
    495.0, 494.0, 493.0, 492.0, 491.0, 490.0, 489.0, 488.0, 487.0, 486.0, 485.0,
    484.0, 483.0, 482.0, 481.0, 480.0, 479.0, 478.0, 477.0, 476.0, 475.0, 474.0,
    473.0, 472.0, 471.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0,
    10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0)
  final private val UPPER_FILTER_COL_C_MANUAL_VALUE = 961.0
  final private val LABEL_COL = "label"
  final private val FILTER_COL = "a"
  final private val EXCLUSION_FIELD = "b"
  final private val MANUAL_FIELD = "c"
  final private val EXCLUSION_COLS = Array("a", "c")
  final private val MANUAL_FILTERS = List(ManualFilters(MANUAL_FIELD, 900.0))

  it should "filter appropriate values from an exponential distribution in 'upper' mode at 95p" in {

    val exponentialData = DiscreteTestDataGenerator.generateOutlierData(
      OUTLIER_ROW_COUNT,
      OUTLIER_LABEL_DISTINCT_COUNT
    )

    val outlierTransformer = new OutlierFiltering(exponentialData)
      .setLabelCol(LABEL_COL)
      .setFilterBounds("upper")
      .setUpperFilterNTile(0.95)
      .setLowerFilterNTile(0.4)
      .setFilterPrecision(0.01)
      .setContinuousDataThreshold(100)
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
      nonFilterCount == EXPECTED_PRESERVE_COUNT_UPPER,
      s"rows of non-filtered outlier data in the 95p upper mode."
    )
    assert(
      filterCount == EXPECTED_FILTER_COUNT_UPPER,
      s"rows of outlier filtered data in the 95p upper mode."
    )
    assert(
      filteredColA == UPPER_FILTER_COL_A_VALUE,
      s"for the correct value of col $FILTER_COL row to be filtered out in the 95p upper mode."
    )
  }

  it should "filter appropriate values from an exponential distribution in 'lower' mode at 5p" in {

    val exponentialData = DiscreteTestDataGenerator.generateOutlierData(
      OUTLIER_ROW_COUNT,
      OUTLIER_LABEL_DISTINCT_COUNT
    )

    val outlierTransformer = new OutlierFiltering(exponentialData)
      .setLabelCol("label")
      .setFilterBounds("lower")
      .setUpperFilterNTile(0.95)
      .setLowerFilterNTile(0.05)
      .setFilterPrecision(0.01)
      .setContinuousDataThreshold(100)
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
      nonFilterCount == EXPECTED_PRESERVE_COUNT_LOWER,
      s"rows of non-filtered outlier data in the 95p lower mode."
    )
    assert(
      filterCount == EXPECTED_FILTER_COUNT_LOWER,
      s"rows of outlier filtered data in the 95p lower mode."
    )
    assert(
      filteredColA == LOWER_FILTER_COL_A_VALUE,
      s"for the correct value of col $FILTER_COL row to be filtered out in the 5p lower mode."
    )
  }

  it should "filter appropriate values from an exponential distribution in 'both' mode at 5p and 95p" in {

    val exponentialData = DiscreteTestDataGenerator.generateOutlierData(
      OUTLIER_ROW_COUNT,
      OUTLIER_LABEL_DISTINCT_COUNT
    )

    val outlierTransformer = new OutlierFiltering(exponentialData)
      .setLabelCol("label")
      .setFilterBounds("both")
      .setUpperFilterNTile(0.95)
      .setLowerFilterNTile(0.05)
      .setFilterPrecision(0.01)
      .setContinuousDataThreshold(100)
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

    val exponentialData = DiscreteTestDataGenerator.generateOutlierData(
      OUTLIER_ROW_COUNT,
      OUTLIER_LABEL_DISTINCT_COUNT
    )

    val outlierTransformer = new OutlierFiltering(exponentialData)
      .setLabelCol("label")
      .setFilterBounds("both")
      .setUpperFilterNTile(0.95)
      .setLowerFilterNTile(0.05)
      .setFilterPrecision(0.01)
      .setContinuousDataThreshold(100)
      .setParallelism(1)

    val filteredHigh = outlierTransformer.filterContinuousOutliers(
      Array("label", AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL),
      EXCLUSION_COLS
    )

    val filterCount = filteredHigh._2.count()
    val nonFilterCount = filteredHigh._1.count()
    val filteredColB =
      filteredHigh._2.collect().map(_.getAs[Double](EXCLUSION_FIELD))

    assert(
      nonFilterCount == EXPECTED_PRESERVE_EXCLUSION_MODE,
      s"rows of non-filtered outlier data in the 95p upper mode."
    )
    assert(
      filterCount == EXPECTED_FILTER_EXCLUSION_MODE,
      s"rows of outlier filtered data in the 95p upper mode."
    )
    assert(
      filteredColB.sameElements(BOTH_FILTER_COL_B_VALUE),
      s"for the correct value of col $EXCLUSION_FIELD row to be filtered out in the 95p upper mode."
    )
  }

  it should "filter appropriate values manually in 'upper' mode." in {

    val exponentialData = DiscreteTestDataGenerator.generateOutlierData(
      OUTLIER_ROW_COUNT,
      OUTLIER_LABEL_DISTINCT_COUNT
    )

    val outlierTransformer = new OutlierFiltering(exponentialData)
      .setLabelCol("label")
      .setFilterBounds("upper")
      .setUpperFilterNTile(0.95)
      .setLowerFilterNTile(0.05)
      .setFilterPrecision(0.01)
      .setContinuousDataThreshold(100)
      .setParallelism(1)

    val filteredHigh = outlierTransformer.filterContinuousOutliers(
      MANUAL_FILTERS,
      Array("label", AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    )

    val filterCount = filteredHigh._2.count()
    val nonFilterCount = filteredHigh._1.count()
    val filteredData =
      filteredHigh._2.collect().map(_.getAs[Double](MANUAL_FIELD)).head

    assert(
      nonFilterCount == EXPECTED_PRESERVE_COUNT_MANUAL,
      s"rows of non-filtered outlier data in the 95p upper mode."
    )
    assert(
      filterCount == EXPECTED_FILTER_COUNT_MANUAL,
      s"rows of outlier filtered data in the 95p upper mode."
    )
    assert(
      filteredData == UPPER_FILTER_COL_C_MANUAL_VALUE,
      s"for the correct value of col $MANUAL_FIELD row to be filtered out in manual mode."
    )
  }

}
