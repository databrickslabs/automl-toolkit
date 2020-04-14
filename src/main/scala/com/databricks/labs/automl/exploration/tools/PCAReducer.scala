package com.databricks.labs.automl.exploration.tools

import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, SparseVector}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StringType

/**
  * API wrapper for conducting a 2-component PCA for visualizing a data set's feature relationships in a way that
  * can be readily visualized.  Provides DataFrame export types for both the raw data with PC1 and PC2 values, as well
  * as the eigen vector values.
  */
class PCAReducer() extends SparkSessionWrapper {

  final private val K_VALUE = 2
  final val SI_NAME = "_si"
  final val TEMP_LABEL_NAME = "tempLabel"
  final val VECTOR_INTERNAL_NAME = "preScaledFeatures"
  final val PCA_INTERNAL_ARRAY_NAME = "pcaArrayColumn"
  final val PCA_DIM_1_NAME = "PCA1"
  final val PCA_DIM_2_NAME = "PCA2"
  final val HANDLE_MODE = "keep"
  final val MINMAX_MIN = -1.0
  final val MINMAX_MAX = 1.0
  final val STANDARD_SCALER_MEAN = true
  final val STANDARD_SCALER_STDDEV = true
  final private val ALLOWABLE_SCALERS: Array[String] =
    Array("minMax", "standard", "maxAbs")

  final private val convertVectorToArray: Any => Array[Double] = x => {
    val v = x match {
      case x: SparseVector => x.toDense
      case _               => x.asInstanceOf[DenseVector]
    }
    v.toArray
  }

  final private val vectorConverter = udf(convertVectorToArray)

  var labelColumn: String = "label"
  var featureColumn: String = "feature"
  var pcaFeatureColumn: String = "pcaFeatures"
  var scalerType: String = "minMax"
  var reportMode: String = "original"

  def setLabelColumn(value: String): this.type = {
    labelColumn = value
    this
  }

  def setFeatureColumn(value: String): this.type = {
    featureColumn = value
    this
  }

  def setPcaFeatureColumn(value: String): this.type = {
    pcaFeatureColumn = value
    this
  }

  def setScalerType(value: String): this.type = {
    assert(
      ALLOWABLE_SCALERS.contains(value),
      s"Scaler Type $value is not supported.  Must be one of ${ALLOWABLE_SCALERS.mkString(", ")}"
    )
    scalerType = value
    this
  }

  def withStandardScaling: this.type = {
    scalerType = "standard"
    this
  }

  def withMinMaxScaling: this.type = {
    scalerType = "minMax"
    this
  }

  def withMaxAbsScaling: this.type = {
    scalerType = "maxAbs"
    this
  }

  def withConvertedColumns: this.type = {
    reportMode = "all"
    this
  }

  def withOriginalColumns: this.type = {
    reportMode = "original"
    this
  }

  def getLabelColumn: String = labelColumn
  def getFeatureColumn: String = featureColumn
  def getPcaFeatureColumn: String = pcaFeatureColumn
  def getScalerType: String = scalerType

  private def checkLabelColumn(data: DataFrame): Unit = {

    assert(
      data.schema.names.contains(labelColumn),
      s"The label column specified '${labelColumn} is not present in the" +
        s"schema payload: '${data.schema.names.mkString(", ")}'"
    )

  }

  /**
    * Validation and checking of the label's data structure.  If non numeric (StringType), then encode the column
    * with a StringIndexer
    * @param data A Dataframe to analyze
    * @return The original DataFrame, with StringIndexed label column if required.
    * @author Ben Wilson, Databricks
    * @since 0.7.2
    */
  private def labelEncode(data: DataFrame): DataFrame = {

    checkLabelColumn(data)

    val labelType = data.schema.filter(_.name == labelColumn).head.dataType

    labelType match {
      case StringType => {
        val indexer = new StringIndexer()
          .setInputCol(labelColumn)
          .setOutputCol(TEMP_LABEL_NAME)
        indexer
          .fit(data)
          .transform(data)
          .drop(labelColumn)
          .withColumnRenamed(TEMP_LABEL_NAME, labelColumn)
      }
      case _ => data
    }

  }

  /**
    * Private method for generating the full PCA pipeline in order to generate the dimensionality reduction.
    * @param data A Dataframe to analyze
    * @return Pipeline object that contains all of the transformation, vectorization, and PCA reduction through SVD.
    * @since 0.7.2
    * @author Ben Wilson, Databricks
    */
  private def createPipeline(data: DataFrame): Pipeline = {

    val labelFormattedData = labelEncode(data)

    val inputSchema = labelFormattedData.schema

    val fullSchemaWithoutLabel = inputSchema.names.filterNot(Set(labelColumn))

    val indexerCandidates =
      inputSchema.filter(_.dataType == StringType).map(_.name)

    val indexers = indexerCandidates
      .map(x => {
        new StringIndexer()
          .setInputCol(x)
          .setOutputCol(x + SI_NAME)
          .setHandleInvalid(HANDLE_MODE)
      })
      .toArray

    val indexerColumns = indexers.map(_.getOutputCol)

    val otherColumns =
      fullSchemaWithoutLabel.filterNot(indexerCandidates.toSet)

    val vectorAssembler =
      new VectorAssembler()
        .setInputCols(otherColumns ++ indexerColumns)
        .setOutputCol(VECTOR_INTERNAL_NAME)

    val preScaledStages = indexers ++ Array(vectorAssembler)

    val allStages = preScaledStages ++ Array(getScalerType match {
      case "minMax" =>
        new MinMaxScaler()
          .setInputCol(VECTOR_INTERNAL_NAME)
          .setOutputCol(featureColumn)
          .setMin(MINMAX_MIN)
          .setMax(MINMAX_MAX)
      case "maxAbs" =>
        new MaxAbsScaler()
          .setInputCol(VECTOR_INTERNAL_NAME)
          .setOutputCol(featureColumn)
      case _ =>
        new StandardScaler()
          .setInputCol(VECTOR_INTERNAL_NAME)
          .setOutputCol(featureColumn)
          .setWithMean(STANDARD_SCALER_MEAN)
          .setWithStd(STANDARD_SCALER_STDDEV)
    }) ++ Array(
      new PCA()
        .setInputCol(featureColumn)
        .setOutputCol(pcaFeatureColumn)
        .setK(K_VALUE)
    )

    new Pipeline().setStages(allStages)

  }

  /**
    * Private method for extracting the Eigen Vectors from the pc matrix
    * @param pipeline Spark ml pipeline to get the vector assembler column payload
    * @param pcaMatrix matrix of eigen vectors from the pca model
    * @return Array[PCAEigenResult] that maps each column name to the Eigen Vectors for each prinicple component
    */
  private def generateEigenValues(
    pipeline: PipelineModel,
    pcaMatrix: DenseMatrix
  ): Array[PCACEigenResult] = {

    val pcaCorrelations = pcaMatrix.rowIter.map(x => x.toArray).toArray
    val inputColumns =
      pipeline.stages
        .collect { case a: VectorAssembler => a }
        .last
        .getInputCols
        .map {
          case z if z.endsWith(SI_NAME) => z.dropRight(SI_NAME.length)
        }

    inputColumns
      .zip(pcaCorrelations)
      .map(x => PCACEigenResult(x._1, x._2.head, x._2.last))

  }

  private def generateEigenDataFrame(
    pcaEigenData: Array[PCACEigenResult]
  ): DataFrame = {

    spark.createDataFrame(pcaEigenData)

  }

  /**
    * Main Method for getting a two dimensional PCA analysis for charting of the principle components in a scatter plot
    * and to get the explained variances and the principle components matrix
    * @param data raw dataframe to analyze
    * @return PCAReducerResult, consisting of the 2 dimensional principle components of the data, the explained variances,
    *         and the PC matrix
    * @author Ben Wilson, Databricks
    * @since 0.7.2
    */
  def executePipeline(data: DataFrame): PCAReducerResult = {

    val originalSchema = data.schema.names

    val pipeline = createPipeline(data)
    val fitted = pipeline.fit(data)

    val fitModel = fitted.stages.last.asInstanceOf[PCAModel]

    val explainedVariances = fitModel.explainedVariance.values

    val pcMatrix = generateEigenValues(fitted, fitModel.pc)

    val pcaDataInitial = pipeline
      .fit(data)
      .transform(data)
      .withColumn(
        PCA_INTERNAL_ARRAY_NAME,
        vectorConverter(col(pcaFeatureColumn))
      )
      .withColumn(PCA_DIM_1_NAME, element_at(col(PCA_INTERNAL_ARRAY_NAME), 1))
      .withColumn(PCA_DIM_2_NAME, element_at(col(PCA_INTERNAL_ARRAY_NAME), 2))

    val pcaData = reportMode match {
      case "original" =>
        pcaDataInitial.select(
          originalSchema.head,
          originalSchema.tail ++ Array(PCA_DIM_1_NAME, PCA_DIM_2_NAME): _*
        )
      case "all" => pcaDataInitial
      case _ =>
        throw new UnsupportedOperationException(
          s"PCA report mode $reportMode is not supported."
        )
    }

    PCAReducerResult(
      pcaData,
      explainedVariances,
      pcMatrix,
      generateEigenDataFrame(pcMatrix)
    )
  }

}

object PCAReducer {
  def apply(labelCol: String,
            featureCol: String,
            pcaFeatureCol: String,
            scalerType: String = "minMax"): PCAReducer =
    new PCAReducer()
      .setLabelColumn(labelCol)
      .setFeatureColumn(featureCol)
      .setPcaFeatureColumn(pcaFeatureCol)
      .setScalerType(scalerType)
}
