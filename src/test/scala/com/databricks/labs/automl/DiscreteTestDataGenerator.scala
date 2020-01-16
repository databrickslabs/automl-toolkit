package com.databricks.labs.automl

import com.databricks.labs.automl.utilities.{
  CardinalityFilteringTestSchema,
  DataGeneratorUtilities,
  FeatureCorrelationTestSchema,
  FeatureInteractionSchema,
  KSampleSchema,
  ModelDetectionSchema,
  NaFillTestSchema,
  OutlierTestSchema,
  PearsonRegressionTestSchema,
  PearsonTestSchema,
  SanitizerSchema,
  SanitizerSchemaRegressor,
  VarianceTestSchema
}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object DiscreteTestDataGenerator extends DataGeneratorUtilities {

  final private val MLFLOWID_START = 1
  final private val MLFLOWID_STEP = 1
  final private val MLFLOW_ID_MODE = "ascending"

  private def generateMlFlowID(rows: Int): Array[Int] = {
    generateRepeatingIntData(
      rows,
      MLFLOWID_START,
      MLFLOWID_STEP,
      MLFLOW_ID_MODE,
      rows
    )
  }

  def generateNAFillData(rows: Int, naRate: Int): DataFrame = {

    val spark = AutomationUnitTestsUtil.sparkSession

    val DOUBLES_START = 1.0
    val DOUBLES_STEP = 2.0
    val DOUBLES_MODE = "ascending"
    val FLOAT_START = 0.0f
    val FLOAT_STEP = 2.0f
    val FLOAT_MODE = "descending"
    val INT_START = 1
    val INT_STEP = 2
    val INT_MODE = "ascending"
    val ORD_START = 5
    val ORD_STEP = 4
    val ORD_MODE = "descending"
    val ORD_DISTINCT_COUNT = 6
    val STRING_DISTINCT_COUNT = 5
    val DATE_YEAR = 2019
    val DATE_MONTH = 7
    val DATE_DAY = 25
    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "random"
    val LABEL_DISTINCT_COUNT = 4

    import spark.implicits._

    val targetNaModulus = rows / naRate

    val doublesSpace = generateDoublesDataWithNulls(
      rows,
      DOUBLES_START,
      DOUBLES_STEP,
      DOUBLES_MODE,
      targetNaModulus,
      0
    )

    val floatSpace = generateFloatsDataWithNulls(
      rows,
      FLOAT_START,
      FLOAT_STEP,
      FLOAT_MODE,
      targetNaModulus,
      1
    )

    val intSpace = generateIntDataWithNulls(
      rows,
      INT_START,
      INT_STEP,
      INT_MODE,
      targetNaModulus,
      2
    )
    val ordinalIntSpace = generateRepeatingIntDataWithNulls(
      rows,
      ORD_START,
      ORD_STEP,
      ORD_MODE,
      ORD_DISTINCT_COUNT,
      targetNaModulus,
      3
    )
    val stringSpace =
      generateStringDataWithNulls(
        rows,
        STRING_DISTINCT_COUNT,
        targetNaModulus,
        4
      )
    val booleanSpace = generateBooleanDataWithNulls(rows, targetNaModulus, 5)
    val daysSpace = generateDatesWithNulls(
      rows,
      DATE_YEAR,
      DATE_MONTH,
      DATE_DAY,
      targetNaModulus,
      6
    )
    val labelData = generateRepeatingIntData(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      LABEL_DISTINCT_COUNT
    )
    val mlFlowIdData = generateMlFlowID(rows)

    val seqData = for (i <- 0 until rows)
      yield
        NaFillTestSchema(
          doublesSpace(i),
          floatSpace(i),
          intSpace(i),
          ordinalIntSpace(i),
          stringSpace(i),
          booleanSpace(i),
          daysSpace(i),
          labelData(i),
          mlFlowIdData(i)
        )
    val dfConversion = seqData
      .toDF()
      .withColumn("dateData", to_date(col("dateData"), "yyyy-MM-dd"))

    reassignToNulls(dfConversion)

  }

  def generateModelDetectionData(rows: Int, uniqueLabels: Int): DataFrame = {

    val FEATURE_START = 2.0
    val FEATURE_STEP = 1.0
    val FEATURE_MODE = "ascending"
    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "random"

    val spark = AutomationUnitTestsUtil.sparkSession

    val featureData =
      generateDoublesData(rows, FEATURE_START, FEATURE_STEP, FEATURE_MODE)
    val labelData = generateRepeatingIntData(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      uniqueLabels
    ).map(_.toDouble)

    val mlFlowIdData = generateMlFlowID(rows)

    val seqData = for (i <- 0 until rows)
      yield ModelDetectionSchema(featureData(i), labelData(i), mlFlowIdData(i))

    import spark.implicits._

    seqData.toDF()

  }

  def generateOutlierData(rows: Int, uniqueLabels: Int): DataFrame = {

    val EXPONENTIAL_START = 0.0
    val EXPONENTIAL_STEP = 2.0
    val EXPONENTIAL_MODE = "ascending"
    val EXPONENTIAL_POWER = 3
    val LINEAR_START = 0.0
    val LINEAR_STEP = 1.0
    val LINEAR_MODE = "descending"
    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "random"
    val LABEL_DISTINCT = 5
    val EXPONENTIAL_TAIL_START = 1.0
    val EXPONENTIAL_TAIL_STEP = 5.0
    val EXPONENTIAL_TAIL_MODE = "ascending"
    val EXPONENTIAL_TAIL_POWER = 2

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._
    val a = generateExponentialData(
      rows,
      EXPONENTIAL_START,
      EXPONENTIAL_STEP,
      EXPONENTIAL_MODE,
      EXPONENTIAL_POWER
    )
    val b = generateDoublesData(rows, LINEAR_START, LINEAR_STEP, LINEAR_MODE)
    val c = generateTailedExponentialData(
      rows,
      EXPONENTIAL_TAIL_START,
      EXPONENTIAL_TAIL_STEP,
      EXPONENTIAL_TAIL_MODE,
      EXPONENTIAL_TAIL_POWER
    )
    val label = generateRepeatingIntData(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      LABEL_DISTINCT
    )
    val mlFlowIdData = generateMlFlowID(rows)

    val seqData = for (i <- 0 until rows)
      yield OutlierTestSchema(a(i), b(i), c(i), label(i), mlFlowIdData(i))

    seqData.toDF()

  }

  def generateVarianceFilteringData(rows: Int): DataFrame = {

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    val DOUBLE_SERIES_START = 1.0
    val DOUBLE_SERIES_STEP = 1.0
    val DOUBLE_SERIES_MODE = "ascending"
    val REPEATING_DOUBLE_VALUE = 42.42
    val REPEATING_INT_VALUE = 9
    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "random"
    val LABEL_DISTINCT_COUNT = 7

    val a = generateDoublesData(
      rows,
      DOUBLE_SERIES_START,
      DOUBLE_SERIES_STEP,
      DOUBLE_SERIES_MODE
    )
    val b = generateFibonacciData(rows)
    val c = generateStaticDoubleSeries(rows, REPEATING_DOUBLE_VALUE)
    val d = generateStaticIntSeries(rows, REPEATING_INT_VALUE)
    val label = generateRepeatingIntData(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      LABEL_DISTINCT_COUNT
    )

    val mlFlowIdData = generateMlFlowID(rows)

    val seqData = for (i <- 0 until rows)
      yield
        VarianceTestSchema(a(i), b(i), c(i), d(i), label(i), mlFlowIdData(i))

    seqData.toDF()
  }

  def generatePearsonFilteringData(rows: Int): (DataFrame, Array[String]) = {

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    val POSITIVE_CORR_1_START = 1
    val POSITIVE_CORR_1_STEP = 1
    val POSITIVE_CORR_1_MODE = "ascending"
    val POSITIVE_CORR_1_DISTINCT_COUNT = 4

    val POSITIVE_CORR_2_START = 1
    val POSITIVE_CORR_2_STEP = 1
    val POSITIVE_CORR_2_MODE = "descending"
    val POSITIVE_CORR_2_DISTINCT_COUNT = 4

    val NOFILTER_1_START = 1.0
    val NOFILTER_1_STEP = 1.0
    val NOFILTER_1_MODE = "random"

    val NOFILTER_2_START = 1
    val NOFILTER_2_STEP = 1
    val NOFILTER_2_MODE = "random"
    val NOFILTER_2_DISTINCT_COUNT = 7

    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "ascending"
    val LABEL_DISTINCT_COUNT = 4

    val positiveCorr1 = generateRepeatingIntData(
      rows,
      POSITIVE_CORR_1_START,
      POSITIVE_CORR_1_STEP,
      POSITIVE_CORR_1_MODE,
      POSITIVE_CORR_1_DISTINCT_COUNT
    )
    val positiveCorr2 = generateRepeatingIntData(
      rows,
      POSITIVE_CORR_2_START,
      POSITIVE_CORR_2_STEP,
      POSITIVE_CORR_2_MODE,
      POSITIVE_CORR_2_DISTINCT_COUNT
    )
    val noFilter1 = generateDoublesData(
      rows,
      NOFILTER_1_START,
      NOFILTER_1_STEP,
      NOFILTER_1_MODE
    )
    val noFilter2 = generateRepeatingIntData(
      rows,
      NOFILTER_2_START,
      NOFILTER_2_STEP,
      NOFILTER_2_MODE,
      NOFILTER_2_DISTINCT_COUNT
    )
    val label = generateRepeatingIntData(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      LABEL_DISTINCT_COUNT
    )
    val mlFlowIdData = generateMlFlowID(rows)

    val seqData = for (i <- 0 until rows)
      yield
        PearsonTestSchema(
          positiveCorr1(i),
          positiveCorr2(i),
          noFilter1(i),
          noFilter2(i),
          label(i),
          mlFlowIdData(i)
        )

    val rawData = seqData.toDF()

    val featureCols =
      rawData.schema.names
        .filterNot(x => x.contains("label") || x.contains("automl_internal_id"))

    val assembler =
      new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    (assembler.transform(rawData), featureCols)

  }

  def generatePearsonRegressionFilteringData(
    rows: Int
  ): (DataFrame, Array[String]) = {

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    val POSITIVE_CORR_1_START = 1
    val POSITIVE_CORR_1_STEP = 1
    val POSITIVE_CORR_1_MODE = "ascending"

    val POSITIVE_CORR_2_START = 1
    val POSITIVE_CORR_2_STEP = 1
    val POSITIVE_CORR_2_MODE = "descending"

    val POSITIVE_CORR_3_START = 1
    val POSITIVE_CORR_3_STEP = 2
    val POSITIVE_CORR_3_MODE = "ascending"

    val NOFILTER_1_START = 1.0
    val NOFILTER_1_STEP = 1.0
    val NOFILTER_1_MODE = "random"

    val NOFILTER_2_START = 1
    val NOFILTER_2_STEP = 1
    val NOFILTER_2_MODE = "random"
    val NOFILTER_2_DISTINCT_COUNT = 7

    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "ascending"

    val positiveCorr1 = generateDoublesData(
      rows,
      POSITIVE_CORR_1_START,
      POSITIVE_CORR_1_STEP,
      POSITIVE_CORR_1_MODE
    )
    val positiveCorr2 = generateDoublesData(
      rows,
      POSITIVE_CORR_2_START,
      POSITIVE_CORR_2_STEP,
      POSITIVE_CORR_2_MODE
    ).map(x => x * -1.0)
    val positiveCorr3 = generateIntData(
      rows,
      POSITIVE_CORR_3_START,
      POSITIVE_CORR_3_STEP,
      POSITIVE_CORR_3_MODE
    )
    val noFilter1 = generateDoublesData(
      rows,
      NOFILTER_1_START,
      NOFILTER_1_STEP,
      NOFILTER_1_MODE
    )
    val noFilter2 = generateRepeatingIntData(
      rows,
      NOFILTER_2_START,
      NOFILTER_2_STEP,
      NOFILTER_2_MODE,
      NOFILTER_2_DISTINCT_COUNT
    )
    val label = generateDoublesData(rows, LABEL_START, LABEL_STEP, LABEL_MODE)
    val mlFlowIdData = generateMlFlowID(rows)

    val seqData = for (i <- 0 until rows)
      yield
        PearsonRegressionTestSchema(
          positiveCorr1(i),
          positiveCorr2(i),
          positiveCorr3(i),
          noFilter1(i),
          noFilter2(i),
          label(i),
          mlFlowIdData(i)
        )

    val rawData = seqData.toDF()

    val featureCols =
      rawData.schema.names
        .filterNot(x => x.contains("label") || x.contains("automl_internal_id"))

    val assembler =
      new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    (assembler.transform(rawData), featureCols)

  }

  def generateFeatureCorrelationData(rows: Int): (DataFrame, Array[String]) = {

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    /**
      * A - 100% correlation in linear series
      * B - 100% correlation in reverse ordering
      * C - 2 of 3 correlated
      * D - repeated categorical no correlation
      */
    val A1_START = 1.0
    val A1_STEP = 1.0
    val A1_MODE = "ascending"
    val A2_START = 1.0
    val A2_STEP = 1.0
    val A2_MODE = "ascending"
    val B1_START = 0
    val B1_STEP = 5
    val B1_MODE = "ascending"
    val B2_START = 0
    val B2_STEP = 5
    val B2_MODE = "ascending"
    val C1_START = 1.0
    val C1_STEP = 3.0
    val C1_MODE = "descending"
    val C2_START = 1.0
    val C2_STEP = 3.0
    val C2_MODE = "ascending"
    val C2_DISTINCT_COUNT = 5
    val C3_START = 1.0
    val C3_STEP = 3.0
    val C3_MODE = "descending"
    val C3_DISTINCT_COUNT = 5
    val D1_START = 100L
    val D1_STEP = 50L
    val D1_MODE = "random"
    val D2_START = 10L
    val D2_STEP = 1L
    val D2_MODE = "random"
    val D2_DISTINCT_COUNT = 10
    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "ascending"
    val LABEL_DISTINCT_COUNT = 4

    val a1 = generateDoublesData(rows, A1_START, A1_STEP, A1_MODE)
    val a2 = generateDoublesData(rows, A2_START, A2_STEP, A2_MODE)
    val b1 = generateIntData(rows, B1_START, B1_STEP, B1_MODE)
    val b2 = generateIntData(rows, B2_START, B2_STEP, B2_MODE).map(x => x * -1)
    val c1 = generateDoublesData(rows, C1_START, C1_STEP, C1_MODE)
    val c2 = generateRepeatingDoublesData(
      rows,
      C2_START,
      C2_STEP,
      C2_MODE,
      C2_DISTINCT_COUNT
    )
    val c3 = generateRepeatingDoublesData(
      rows,
      C3_START,
      C3_STEP,
      C3_MODE,
      C3_DISTINCT_COUNT
    )
    val d1 = generateLongData(rows, D1_START, D1_STEP, D1_MODE)
    val d2 = generateRepeatingLongData(
      rows,
      D2_START,
      D2_STEP,
      D2_MODE,
      D2_DISTINCT_COUNT
    )
    val label = generateRepeatingIntData(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      LABEL_DISTINCT_COUNT
    )
    val mlFlowIdData = generateMlFlowID(rows)

    val seqData = for (i <- 0 until rows)
      yield
        FeatureCorrelationTestSchema(
          a1(i),
          a2(i),
          b1(i),
          b2(i),
          c1(i),
          c2(i),
          c3(i),
          d1(i),
          d2(i),
          label(i),
          mlFlowIdData(i)
        )

    val rawData = seqData.toDF()

    val featureCols =
      rawData.schema.names
        .filterNot(x => x.contains("label") || x.contains("automl_internal_id"))

    (rawData, featureCols)

  }

  def generateCardinalityFilteringData(
    rows: Int
  ): (DataFrame, Array[String]) = {

    val spark = AutomationUnitTestsUtil.sparkSession

    val CATEGORICAL_FIELDS = Array("b", "c", "d")

    import spark.implicits._

    val A_START = 1.0
    val A_STEP = 5.0
    val A_MODE = "ascending"
    val B_START = 1
    val B_STEP = 1
    val B_MODE = "ascending"
    val B_DISTINCT_COUNT = 3
    val C_START = 10L
    val C_STEP = 10L
    val C_MODE = "descending"
    val C_DISTINCT_COUNT = 10
    val D_DISTINCT_COUNT = 55

    val a = generateDoublesData(rows, A_START, A_STEP, A_MODE)
    val b =
      generateRepeatingIntData(rows, B_START, B_STEP, B_MODE, B_DISTINCT_COUNT)
    val c =
      generateRepeatingLongData(rows, C_START, C_STEP, C_MODE, C_DISTINCT_COUNT)
    val d = generateStringData(rows, D_DISTINCT_COUNT)

    val seqData = for (i <- 0 until rows)
      yield CardinalityFilteringTestSchema(a(i), b(i), c(i), d(i))
    val data = seqData.toDF()

    (data, CATEGORICAL_FIELDS)

  }

  def generateSanitizerData(rows: Int, modelType: String): DataFrame = {

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    val A_START = 1.0
    val A_STEP = 2.0
    val A_MODE = "descending"
    val B_START = 1
    val B_STEP = 1
    val B_MODE = "ascending"
    val B_DISTINCT_COUNT = 3
    val C_START = 100L
    val C_STEP = 100L
    val C_MODE = "random"
    val C_DISTINCT_COUNT = 500
    val D_DISTINCT_COUNT = 12
    val E_START = 1000
    val E_STEP = 10
    val E_MODE = "ascending"
    val LABEL_DISTINCT_COUNT = 4
    val LABEL_REGRESSION_START = 1.0
    val LABEL_REGRESSION_STEP = 3.0
    val LABEL_REGRESSION_MODE = "ascending"

    val a = generateDoublesData(rows, A_START, A_STEP, A_MODE)
    val b =
      generateRepeatingIntData(rows, B_START, B_STEP, B_MODE, B_DISTINCT_COUNT)
    val c =
      generateRepeatingLongData(rows, C_START, C_STEP, C_MODE, C_DISTINCT_COUNT)
    val d = generateStringData(rows, D_DISTINCT_COUNT)
    val e = generateIntData(rows, E_START, E_STEP, E_MODE)
    val f = generateBooleanData(rows)
    val mlflow = generateMlFlowID(rows)

    val label = generateStringData(rows, LABEL_DISTINCT_COUNT)
    val labelRegression = generateDoublesData(
      rows,
      LABEL_REGRESSION_START,
      LABEL_REGRESSION_STEP,
      LABEL_REGRESSION_MODE
    )

    val output = modelType match {
      case "classifier" =>
        val seqData = for (i <- 0 until rows)
          yield
            SanitizerSchema(
              a(i),
              b(i),
              c(i),
              d(i),
              e(i),
              f(i),
              label(i),
              mlflow(i)
            )
        seqData.toDF()
      case "regressor" =>
        val seqData = for (i <- 0 until rows)
          yield
            SanitizerSchemaRegressor(
              a(i),
              b(i),
              c(i),
              d(i),
              e(i),
              f(i),
              labelRegression(i),
              mlflow(i)
            )
        seqData.toDF()
    }

    output
  }

  def generateKSampleData(rows: Int): DataFrame = {

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    val A_START = 1.0
    val A_STEP = 1.0
    val A_MODE = "ascending"
    val B_START = 1
    val B_STEP = 1
    val B_MODE = "descending"
    val B_DISTINCT_COUNT = 4
    val C_START = 10.0
    val C_STEP = 10.0
    val C_MODE = "ascending"
    val C_DISTINCT_COUNT = 6
    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "ascending"
    val LABEL_DISTINCT_COUNT = 3
    val mlflow = generateMlFlowID(rows)

    val a = generateDoublesData(rows, A_START, A_STEP, A_MODE)
    val b =
      generateRepeatingIntData(rows, B_START, B_STEP, B_MODE, B_DISTINCT_COUNT)
    val c =
      generateDoublesBlocks(rows, C_START, C_STEP, C_MODE, C_DISTINCT_COUNT)
    val label = generateIntegerBlocksSkewed(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      LABEL_DISTINCT_COUNT
    )

    val seqData = for (i <- 0 until rows)
      yield KSampleSchema(a(i), b(i), c(i), label(i), mlflow(i))

    seqData.toDF()

  }

  def generateFeatureInteractionData(rows: Int): DataFrame = {

    val spark = AutomationUnitTestsUtil.sparkSession

    import spark.implicits._

    val A_START = 0.0
    val A_STEP = 0.1
    val A_MODE = "ascending"
    val B_START = 1.0
    val B_STEP = 1.0
    val B_MODE = "ascending"
    val C_START = 1
    val C_STEP = 2
    val C_MODE = "ascending"
    val C_DISTINCT_COUNT = 3
    val D_DISTINCT_COUNT = 9
    val E_START = 1
    val E_STEP = 1
    val E_MODE = "descending"
    val E_DISTINCT_COUNT = 5
    val F_DISTINCT_COUNT = 7
    val LABEL_START = 1
    val LABEL_STEP = 1
    val LABEL_MODE = "ascending"
    val LABEL_DISTINCT_COUNT = 4
    val mlflow = generateMlFlowID(rows)

    val a = generateDoublesData(rows, A_START, A_STEP, A_MODE)
    val b = generateDoublesData(rows, B_START, B_STEP, B_MODE)
    val c =
      generateRepeatingIntData(rows, C_START, C_STEP, C_MODE, C_DISTINCT_COUNT)
    val d = generateStringData(rows, D_DISTINCT_COUNT)
    val e =
      generateRepeatingIntData(rows, E_START, E_STEP, E_MODE, E_DISTINCT_COUNT)
    val f = generateStringData(rows, F_DISTINCT_COUNT)
    val label = generateRepeatingIntData(
      rows,
      LABEL_START,
      LABEL_STEP,
      LABEL_MODE,
      LABEL_DISTINCT_COUNT
    )

    val seqData = for (i <- 0 until rows)
      yield
        FeatureInteractionSchema(
          a(i),
          b(i),
          c(i),
          d(i),
          e(i),
          f(i),
          label(i),
          mlflow(i)
        )

    seqData.toDF()

  }

}
