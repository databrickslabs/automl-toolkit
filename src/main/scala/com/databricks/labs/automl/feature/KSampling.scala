package com.databricks.labs.automl.feature

import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{MaxAbsScaler, MinHashLSH, MinHashLSHModel}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._

import scala.collection.mutable.ListBuffer

trait KSamplingBase extends KSamplingDefaults with SparkSessionWrapper {

  final private val allowableKMeansDistanceMeasurements: List[String] =
    List("cosine", "euclidean")

  private[feature] var _featuresCol: String = defaultFeaturesCol
  private[feature] var _kGroups: Int = defaultKGroups
  private[feature] var _kMeansMaxIter: Int = defaultKMeansMaxIter
  private[feature] var _kMeansTolerance: Double = defaultKMeansTolerance
  private[feature] var _kMeansDistanceMeasurement: String =
    defaultKMeansDistanceMeasurement
  private[feature] var _kMeansSeed: Long = defaultKMeansSeed
  private[feature] var _kMeansPredictionCol: String = defaultKMeansPredictionCol
  private[feature] var _lshHashTables = defaultHashTables
  private[feature] var _lshSeed = defaultLSHSeed
  private[feature] var _lshOutputCol = defaultLSHOutputCol
  private[feature] var _quorumCount = defaultQuorumCount

  private[feature] var conf = getKSamplingConfig

  def setFeaturesCol(value: String): this.type = {
    _featuresCol = value; setConfig; this
  }
  def setKGroups(value: Int): this.type = {
    _kGroups = value
    setConfig
    this
  }
  def setKMeansMaxIter(value: Int): this.type = {
    _kMeansMaxIter = value
    setConfig
    this
  }

  //TODO: range checking
  def setKMeansTolerance(value: Double): this.type = {
    _kMeansTolerance = value
    setConfig
    this
  }
  def setKMeansDistanceMeasurement(value: String): this.type = {
    require(
      allowableKMeansDistanceMeasurements.contains(value),
      s"Kmeans Distance Measurement $value is not " +
        s"a valid mode of operation.  Must be one of: ${allowableKMeansDistanceMeasurements.mkString(", ")}"
    )
    _kMeansDistanceMeasurement = value
    setConfig
    this
  }
  def setKMeansSeed(value: Long): this.type = {
    _kMeansSeed = value
    setConfig
    this
  }
  def setKMeansPredictionCol(value: String): this.type = {
    _kMeansPredictionCol = value; this
  }
  def setLSHHashTables(value: Int): this.type = {
    _lshHashTables = value
    setConfig
    this
  }
  def setLSHOutputCol(value: String): this.type = {
    _lshOutputCol = value
    setConfig
    this
  }
  def setQuorumCount(value: Int): this.type = {
    _quorumCount = value
    setConfig
    this
  }

  private def setConfig: this.type = {
    conf = KSamplingConfiguration(
      featuresCol = _featuresCol,
      kGroups = _kGroups,
      kMeansMaxIter = _kMeansMaxIter,
      kMeansTolerance = _kMeansTolerance,
      kMeansDistanceMeasurement = _kMeansDistanceMeasurement,
      kMeansSeed = _kMeansSeed,
      kMeansPredictionCol = _kMeansPredictionCol,
      lshHashTables = _lshHashTables,
      lshSeed = _lshSeed,
      lshOutputCol = _lshOutputCol,
      quorumCount = _quorumCount
    )
    this
  }

  def getKSamplingConfig: KSamplingConfiguration = {
    KSamplingConfiguration(
      featuresCol = _featuresCol,
      kGroups = _kGroups,
      kMeansMaxIter = _kMeansMaxIter,
      kMeansTolerance = _kMeansTolerance,
      kMeansDistanceMeasurement = _kMeansDistanceMeasurement,
      kMeansSeed = _kMeansSeed,
      kMeansPredictionCol = _kMeansPredictionCol,
      lshHashTables = _lshHashTables,
      lshSeed = _lshSeed,
      lshOutputCol = _lshOutputCol,
      quorumCount = _quorumCount
    )
  }

}

class KSampling extends KSamplingBase {

  /**
    * Build a KMeans model in order to find centroids for data simulation
    * @param data The source DataFrame, consisting of the feature fields and a vector column
    * @return KMeansModel that will be used to extract the centroid vectors
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def buildKMeans(data: DataFrame): KMeansModel = {

    val model = new KMeans()
      .setK(conf.kGroups)
      .setSeed(conf.kMeansSeed)
      .setFeaturesCol(conf.featuresCol)
      .setDistanceMeasure(conf.kMeansDistanceMeasurement)
      .setPredictionCol(conf.kMeansDistanceMeasurement)
      .setTol(conf.kMeansTolerance)
      .setMaxIter(conf.kMeansMaxIter)

    model.fit(data)
  }

  /**
    * Build a MinHashLSH Model so that approximate nearest neighbors can be used to find adjacent vectors to a given
    * centroid vector
    * @param data The source DataFrame, consisting of the feature fields and a vector column
    * @return MinHashLSHModel that will be used to generate distances between centroids and a provided vector
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def buildLSH(data: DataFrame): MinHashLSHModel = {

    val model = new MinHashLSH()
      .setNumHashTables(conf.lshHashTables)
      .setSeed(conf.lshSeed)
      .setInputCol(conf.featuresCol)
      .setOutputCol(conf.lshOutputCol)

    model.fit(data)
  }

  /**
    * Method for scaling the feature vector to enable better K Means performance for highly unbalanced vectors
    * @param data The 'raw' vector assembled dataframe
    * @return A DataFrame that has the feature vector scaled as a replacement.
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def scaleFeatureVector(data: DataFrame): DataFrame = {

    val renamedCol = conf.featuresCol + "_f"

    val renamedData = data.withColumnRenamed(conf.featuresCol, renamedCol)

    // Initialize the Scaler
    val scaler = new MaxAbsScaler()
      .setInputCol(renamedCol)
      .setOutputCol(conf.featuresCol)

    // Create the scaler model
    val scalerModel = scaler.fit(renamedData)

    // Apply the scaler and replace the feature vector with scaled features
    scalerModel.transform(renamedData).drop(renamedCol)

  }

  /**
    * Method for getting representative rows that are closest to the calculated centroid positions.
    *
    * @param data The DataFrame that has been transformed by the LSH Model
    * @param lshModel a fit MinHashLSHModel
    * @param kModel a fit KMeansModel
    * @return Array of CentroidVectors that contains the vector and the KMeans Group that it is assigned to.
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def acquireNearestVectorToCentroids(
    data: DataFrame,
    lshModel: MinHashLSHModel,
    kModel: KMeansModel
  ): Array[CentroidVectors] = {

    val centerCandidates = kModel.clusterCenters
      .map { x =>
        lshModel
          .approxNearestNeighbors(kModel.transform(data), x, conf.quorumCount)
          .toDF
      }
      .reduce(_ union _)
      .distinct
      .withColumn(
        "rank",
        dense_rank.over(
          Window
            .partitionBy(col(conf.kMeansPredictionCol))
            .orderBy(col("distCol"), col(conf.featuresCol))
        )
      )
      .where(col("rank") === 1)
      .drop("rank")

    centerCandidates
      .select(col(conf.featuresCol), col(conf.kMeansPredictionCol))
      .collect()
      .map { x =>
        CentroidVectors(
          x.getAs[org.apache.spark.ml.linalg.Vector](conf.featuresCol),
          x.getAs[Int](conf.kMeansPredictionCol)
        )
      }

  }

  /**
    * Method for retrieving the feature vectors that are closest in n-dimensional space to the provided vector
    * @param data The transformed data (transformed from KMeans)
    * @param lshModel a fit MinHashLSHModel
    * @param vectorCenter The vector and the kGroup that is closest to that group's centroid
    * @param targetCount The desired number of closest neighbors to find
    * @return A DataFrame consisting of the rows that are closest to the supplied vector nearest the centroid
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def acquireNeighborVectors(data: DataFrame,
                                              lshModel: MinHashLSHModel,
                                              vectorCenter: CentroidVectors,
                                              targetCount: Int): DataFrame = {

    lshModel
      .approxNearestNeighbors(
        data.filter(col("kGroups") === vectorCenter.kGroup),
        vectorCenter.vector,
        targetCount
      )
      .toDF

  }

  def getRowAsMap(row: org.apache.spark.sql.Row): Map[String, Any] = {
    val rowSchema = row.schema.fieldNames
    row.getValuesMap[Any](rowSchema)
  }

  def mutateValueFixed(first: Double,
                       second: Double,
                       mutationValue: Double): Double = {

    val minVal = scala.math.min(first, second)
    val maxVal = scala.math.max(first, second)
    (minVal * mutationValue) + (maxVal * (1 - mutationValue))
  }

  def ratioValueFixed(first: Double,
                      second: Double,
                      mutationValue: Double): Double = {
    (first + second) * mutationValue
  }

  def mutateValueRandom(first: Double, second: Double): Double = {
    val rand = scala.util.Random
    mutateValueFixed(first, second, rand.nextDouble())
  }

  def toDoubleType(x: Any): Option[Double] = x match {
    case i: Int    => Some(i)
    case d: Double => Some(d)
    case _         => None
  }

  def mutateRow(originalRow: Map[String, Any],
                mutateRow: Map[String, Any],
                indexMapping: Array[RowMapping],
                indexesToMutate: List[Int],
                mode: String,
                mutation: Double): List[Double] = {

    val outputRow = ListBuffer[Double]()

    indexMapping.foreach { r =>
      val origData = originalRow(r.fieldName).toString.toDouble
      val mutateData = mutateRow(r.fieldName).toString.toDouble

      if (indexesToMutate.contains(r.idx)) {

        mode match {
          case "weighted" =>
            outputRow += mutateValueFixed(origData, mutateData, mutation)
          case "random" => outputRow += mutateValueRandom(origData, mutateData)
          case "ratio" =>
            outputRow += ratioValueFixed(origData, mutateData, mutation)
        }

      } else outputRow += origData

    }

    outputRow.toList

  }

  def generateRandomIndexPositions(vectorSize: Int,
                                   minimumCount: Int): List[Int] = {
    val candidateList = List.range(0, vectorSize)
    val restrictionSize = scala.util.Random.nextInt(vectorSize)
    val adjustedSize =
      if (restrictionSize < minimumCount) minimumCount else restrictionSize
    scala.util.Random.shuffle(candidateList).take(adjustedSize).sortWith(_ < _)
  }

  def acquireRowCollections(nearestNeighborData: DataFrame,
                            targetCount: Int,
                            minVectorsToMutate: Int,
                            mutationMode: String,
                            mutationValue: Double): List[List[Double]] = {

    val mutatedRows = ListBuffer[List[Double]]()

    val colIdx = nearestNeighborData.schema.names.zipWithIndex
      .map(x => RowMapping.tupled(x))

    val kGroupCollection = nearestNeighborData.collect.map(getRowAsMap)

    val (center, others) = kGroupCollection.splitAt(1)

    val centerVector = center(0)

    val vectorLength = centerVector.size
    val candidateLength = others.length

    var iter = 0
    var idx = 0

    val minIndexes =
      if (minVectorsToMutate > vectorLength) vectorLength
      else minVectorsToMutate

    do {

      val indexesToMutate =
        generateRandomIndexPositions(vectorLength, minIndexes)

      mutatedRows += mutateRow(
        others(idx),
        centerVector,
        colIdx,
        indexesToMutate,
        mutationMode,
        mutationValue
      )

      iter += 1
      if (idx >= candidateLength - 1) idx = 0 else idx += 1

    } while (iter < targetCount)

    mutatedRows.toList

  }

  def generateDoublesSchema(data: DataFrame,
                            fieldsToExclude: List[String]): StructType = {

    val baseStruct = new StructType()

    val baseSchema = data
      .drop(fieldsToExclude: _*)
      .schema
      .names
      .flatMap(x => baseStruct.add(x, DoubleType, nullable = false))

    DataTypes.createStructType(baseSchema)
  }

  def convertCollectionsToDataFrame(collections: List[List[Double]],
                                    schema: StructType): DataFrame = {
    spark.createDataFrame(sc.makeRDD(collections.map(x => Row(x: _*))), schema)
  }

  def generateGroupVectors[T: scala.math.Numeric](
    clusteredData: DataFrame,
    centroids: Array[CentroidVectors],
    lshModel: MinHashLSHModel,
    labelCol: String,
    labelGroup: T,
    targetCount: Int,
    fieldsToDrop: List[String]
  ): DataFrame = {

    val rowsToGeneratePerGroup =
      scala.math.ceil(targetCount / centroids.length).toInt

    centroids
      .map { x =>
        acquireNeighborVectors(
          clusteredData.filter(col(labelCol) === labelGroup),
          lshModel,
          x,
          rowsToGeneratePerGroup
        )
      }
      .reduce(_.union(_))
      .drop(fieldsToDrop: _*)
      .limit(targetCount)

  }

  def sparkToScalaTypeConversion(sparkType: DataType): String = {

    sparkType match {
      case x: ByteType      => "Byte"
      case x: ShortType     => "Short"
      case x: LongType      => "Long"
      case x: FloatType     => "Float"
      case x: StringType    => "String"
      case x: BinaryType    => "Array[Byte]"
      case x: BooleanType   => "Boolean"
      case x: TimestampType => "java.sql.Timestamp"
      case x: DateType      => "java.sql.Date"
      case x: IntegerType   => "Int"
      case x: DoubleType    => "Double"
      case _                => "Unknown"
    }

  }

  def scalaToSparkTypeConversion(scalaType: String): DataType = {
    scalaType match {
      case "Byte"               => ByteType
      case "Short"              => ShortType
      case "Long"               => LongType
      case "Float"              => FloatType
      case "String"             => StringType
      case "Array[Byte]"        => BinaryType
      case "Boolean"            => BooleanType
      case "java.sql.Timestamp" => TimestampType
      case "java.sql.Date"      => DateType
      case "Int"                => IntegerType
      case "Double"             => DoubleType
    }
  }

  def generateSchemaInformationPayload(
    fullSchema: StructType,
    fieldsNotInVector: Array[String]
  ): SchemaDefinitions = {

    val schemaMapped: Seq[StructMapping] =
      fullSchema.zipWithIndex.map(x => StructMapping.tupled(x))

    // Extract the schema as a case class collection for further manipulations
    val allFields: Seq[SchemaMapping] = schemaMapped.map { x =>
      SchemaMapping(
        fieldName = x.field.name,
        originalFieldIndex = x.idx,
        dfType = x.field.dataType,
        scalaType = sparkToScalaTypeConversion(x.field.dataType)
      )
    }

    // Get only the fields involved in the feature vector
    val featureFields: Seq[RowMapping] = schemaMapped
      .filterNot(x => fieldsNotInVector.contains(x.field.name))
      .map(x => RowMapping(x.field.name, x.idx))

    SchemaDefinitions(allFields.toArray, featureFields.toArray)

  }

  /**
    * Method for converting the feature fields back to the correct types so that the DataFrames can be unioned together.
    */
  def castColumnsToCorrectTypes(dataFrame: DataFrame,
                                schemaPayload: SchemaDefinitions): DataFrame = {

    // Extract the fields and types that are part of the feature vector.
    val featureFieldsPayload = schemaPayload.features
      .map(x => x.fieldName)
      .flatMap(
        y => schemaPayload.fullSchema.filter(z => z.fieldName.contains(y))
      )

    // Perform casting by applying the original DataTypes to the feature vector fields.
    featureFieldsPayload.foldLeft(dataFrame) {
      case (accum, x) =>
        accum.withColumn(x.fieldName, dataFrame(x.fieldName).cast(x.dfType))
    }

  }

  /**
    *  Rebuild the Dataframe so that the fields can be matched for a union.
    *
    */
  def fillMissingColumns(dataFrame: DataFrame,
                         schemaPayload: SchemaDefinitions,
                         featureCol: String,
                         labelCol: String): DataFrame = {

    // Get a roster of all current fields
    val currentSchema = dataFrame.schema.names ++ featureCol ++ labelCol

    // Find the fields that don't exist yet and remove the feature and label columns from the manifest.
    val fieldsToAdd = schemaPayload.fullSchema
      .map(x => x.fieldName)
      .filterNot(y => currentSchema.contains(y))

    // Get the definition of the fields that need to be added.
    val fieldsToAddDefinition =
      schemaPayload.fullSchema.filter(x => fieldsToAdd.contains(x.fieldName))

    fieldsToAddDefinition.foldLeft(dataFrame) {
      case (accum, x) =>
        accum.withColumn(x.fieldName, lit(x.dfType match {
          case _: IntegerType => defaultFill(x.dfType).toString.toInt
          case _: DoubleType  => defaultFill(x.dfType).toString.toDouble
          // TODO: fill out the rest of this.  And split this out into its own method.
        }))
    }

  }
  /**
  *
  *
  *
  * TODO: Finish this logic
  *
  * //TODO: then finish the conversion of the resulting DF to the source schema, adding in the required fields.
  *
  * val kConf = ClusteringConfiguration(k =20, featuresCol = "features", maxIter = 5000, tolerance = 1E-10, distanceMeasure = "euclidean", modelSeed = 42L, predictionCol = "kGroups")
  * val lshConf = LSHConfiguration(numHashTables = 5, modelSeed = 42L, inputCol = "features", outputCol = "hashes")
  * val synConf = SyntheticDataConfig(kConf, lshConf)
  *
  * val featNew = scaleFeatureVector(featRaw, "features")
  * val kModS = buildKMeans(featNew, kConf)
  * val kDataS = kModS.transform(featNew)
  * val lshModS = buildLSH(featNew, lshConf)
  * val vecsS = acquireNearestVectorToCentroids(featNew.filter(col("income")===1), lshModS, kModS, synConf, 11)
  *
  * val tt2 = generateGroupVectors(kDataS, vecsS, lshModS, "income", 1.0, 5000, List("distCol", "hashes"))
  *
  *
  * val test = generateSchemaInformationPayload(featSchema, Array("income", "features"))
  * val ff = test.features.map(x => x.fieldName).map(x => test.fullSchema.filter(y => y.fieldName.contains(x))).flatten
  *
  * val tdf = ff.foldLeft(featRaw) { case (accum, x) => accum.withColumn(x.fieldName, featRaw(x.fieldName).cast(IntegerType))}
  * val tds = ff.foldLeft(tdf) { case (accum, x) => accum.withColumn(x.fieldName, tdf(x.fieldName).cast(x.dfType))}.drop("marital_status_si")
  *
  */

}

object KSampling extends KSamplingBase {}
