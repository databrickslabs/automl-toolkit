package com.databricks.labs.automl.feature

import java.util.{Calendar, Date}

import com.databricks.labs.automl.feature.structures.{
  CentroidVectors,
  RowGenerationConfig,
  RowMapping,
  SchemaDefinitions,
  SchemaMapping,
  StructMapping
}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{
  MaxAbsScaler,
  MinHashLSH,
  MinHashLSHModel,
  VectorAssembler
}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, Row}

import scala.collection.mutable.ListBuffer

class KSampling(df: DataFrame) extends KSamplingBase {

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
      .setPredictionCol(conf.kMeansPredictionCol)
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
        data.filter(col(conf.kMeansPredictionCol) === vectorCenter.kGroup),
        vectorCenter.vector,
        targetCount
      )
      .toDF

  }

  /**
    * Method for converting the Row object to a map of key/value pairs
    *
    * @param row a row of data
    * @return a Map of key/value pairs for the feature columns of the row.
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def getRowAsMap(
    row: org.apache.spark.sql.Row
  ): Map[String, Any] = {
    val rowSchema = row.schema.fieldNames
    row.getValuesMap[Any](rowSchema)
  }

  /**
    * Method for mutating between two row values of all features in the rows.
    * A mutation value is set to provide a ratio between the min and max values.
    * @param first a value to mix with the second variable
    * @param second a value to mix with the first variable
    * @param mutationValue the ratio of mixing between the two variables.
    * @return The scaled value between first and second
    * @author Ben Wilson
    * @since 0.5.1
    */
  def mutateValueFixed(first: Double,
                       second: Double,
                       mutationValue: Double): Double = {

    val minVal = scala.math.min(first, second)
    val maxVal = scala.math.max(first, second)
    (minVal * mutationValue) + (maxVal * (1 - mutationValue))
  }

  /**
    * Method for mutating between row values with a fixed ratio value
    * @param first a value to mix
    * @param second a value to mix
    * @param mutationValue ratio modifier between the two values
    * @return the mutated value
    * @author Ben Wilson
    * @since 0.5.1
    */
  def ratioValueFixed(first: Double,
                      second: Double,
                      mutationValue: Double): Double = {
    (first + second) * mutationValue
  }

  /**
    * Method for randomly mutating between the bounds of two values
    * @param first a value to mix
    * @param second a value to mix
    * @return the randomly mutated value
    * @author Ben Wilson
    * @since 0.5.1
    */
  def mutateValueRandom(first: Double, second: Double): Double = {
    val rand = scala.util.Random
    mutateValueFixed(first, second, rand.nextDouble())
  }

  /**
    * Method for converting Integer Types to Double Types
    * @param x Numeric: Integer or Double
    * @return Option Double type conversion
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def toDoubleType(x: Any): Option[Double] = x match {
    case i: Int    => Some(i)
    case f: Float  => Some(f)
    case l: Long   => Some(l)
    case d: Double => Some(d)
    case _         => None
  }

  /**
    * Method for modifying a row of feature data by finding a linear point along the
    * vector between the features.
    * Options for modification of a vector include:
    * - Full Vector mutation based on a provided ratio, random modification, or a weighted average
    * - Partial Random Vector mutation of a random number of index sites, bound by a lower limit
    * - Fixed Vector mutation of random indexes at a constant count of indexes to modify
    * @param originalRow The Map() representation of one of the vectors
    * @param mutateRow The Map() representation of the other vector
    * @param indexMapping Vector definition of the payload (field name and index)
    * @param indexesToMutate A list of Integers of index positions within the vector to mutate
    * @param mode The method of mutation (weighted average, ratio, or random mutation)
    * @param mutation The magnitude of % share of the value from the centroid-adjacent vector
    *                 to the other vector.  A higher percentage value will be closer in euclidean
    *                 distance to the centroid vector.
    * @return A List of Doubles of the feature vector modifications (data for a new row)
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def mutateRow(originalRow: Map[String, Any],
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

  /**
    * Method for generating a random collection of random-counts of vectors for mutation
    * @param vectorSize The size of the feature vector components (columns)
    * @param minimumCount The minimum number of vectors that can be selected for mutation
    *                     - used to ensure that at least some values will be modified.
    * @return The list of indexes that will be modified through averaging along the vector between
    *         the centroid and a chosen feature vector
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def generateRandomIndexPositions(
    vectorSize: Int,
    minimumCount: Int
  ): List[Int] = {
    val candidateList = List.range(0, vectorSize)
    val restrictionSize = scala.util.Random.nextInt(vectorSize)
    val adjustedSize =
      if (restrictionSize < minimumCount) minimumCount else restrictionSize
    scala.util.Random.shuffle(candidateList).take(adjustedSize).sortWith(_ < _)
  }

  /**
    * Method for generating a fixed number of random indexes in the feature vector to manipulate.
    * @param vectorSize The size of the feature vector
    * @param minimumCount The number of features to mutate which are randomly selected.
    * @note if the value specified in .setMinimumVectorCountToMutate is greater than the
    *       feature vector size, all indexes will be mutated.
    * @return The list of indexes selected for mutation
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def generateFixedIndexPositions(
    vectorSize: Int,
    minimumCount: Int
  ): List[Int] = {
    val candidateList = List.range(0, vectorSize)
    val adjustedSize =
      if (minimumCount > vectorSize) vectorSize else minimumCount
    scala.util.Random.shuffle(candidateList).take(adjustedSize).sortWith(_ < _)
  }

  /**
    * Builder method for controlling what type of index selection will be used, returning the list
    * of indexes that are selected for mutation
    * @param vectorSize The size of the feature vector
    * @return The list of indexes selected for mutation
    * @author Ben Wilson
    * @since 0.5.1
    * @throws IllegalArgumentException() if an invalid entry is made.
    */
  @throws(classOf[IllegalArgumentException])
  private[feature] def generateIndexPositions(vectorSize: Int): List[Int] = {

    conf.vectorMutationMethod match {
      case "random" =>
        generateRandomIndexPositions(
          vectorSize,
          conf.minimumVectorCountToMutate
        )
      case "all" => List.range(0, vectorSize)
      case "fixed" =>
        generateFixedIndexPositions(vectorSize, conf.minimumVectorCountToMutate)
      case _ =>
        throw new IllegalArgumentException(
          s"Vector Mutation Method ${conf.vectorMutationMethod} is not supported.  " +
            s"Please use one of: ${allowableVectorMutationMethods.mkString(", ")}"
        )
    }

  }

  /**
    * Method for generating a collection of synthetic row data, stored as lists of lists of doubles.
    * @param nearestNeighborData A Dataframe that has been transformed by a KMeans model
    * @param targetCount The target number of synthetic rows to generate
    * @param minVectorsToMutate The minimum (or exact) target of vectors to mutate in the feature vector
    * @param mutationMode The mutation mode (random, weighted, or Ratio)
    * @param mutationValue The value of vector ratio share between the centroid-associated vector and the other vectors
    * @return A list of Lists of Doubles (basic DF structure)
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def acquireRowCollections(
    nearestNeighborData: DataFrame,
    targetCount: Int,
    minVectorsToMutate: Int,
    mutationMode: String,
    mutationValue: Double
  ): List[List[Double]] = {

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

      val indexesToMutate = generateIndexPositions(vectorLength)

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

  /**
    * Helper method for constructing a valid DF schema Struct object wherein all of the numeric columns types are
    * converted to DoubleType
    * @param data The DataFrame that contains various numeric type data columns
    * @param fieldsToExclude Fields that should not be included in the conversion and subsequent schema definition
    * @note This function allows for casting the mutated return value of acquireRowCollections to a DataFrame since
    *       the return types of that collection are all of DoubleType.
    * @return A DoubleType encoded Schema
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def generateDoublesSchema(
    data: DataFrame,
    fieldsToExclude: List[String]
  ): StructType = {

    val baseStruct = new StructType()

    val baseSchema = data
      .drop(fieldsToExclude: _*)
      .schema
      .names
      .flatMap(x => baseStruct.add(x, DoubleType, nullable = false))

    DataTypes.createStructType(baseSchema)
  }

  /**
    * Method for converting the raw `List[List[Double]]` collection from the data generator method acquireRowCollections
    * to a DataFrame object
    * @param collections The collection of synthetic feature data in `List[List[Double]]` format
    * @param schema The Double-formatted schema from the helper method generateDoublesSchema
    * @return A DataFrame that has all numeric types of feature columns as DoubleType.
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def convertCollectionsToDataFrame(
    collections: List[List[Double]],
    schema: StructType
  ): DataFrame = {
    spark.createDataFrame(sc.makeRDD(collections.map(x => Row(x: _*))), schema)
  }

  /**
    * Method for generating the Group Vector Centroids that are used for generating the mutated feature rows.
    * @param clusteredData KMeans transformed DataFrame
    * @param centroids Array of the Centroid Vector data for look-up purposes through MinHashLSH
    * @param lshModel The trained MinHashLSH Model
    * @param labelCol The label Column of the DataFrame
    * @param labelGroup The canonical class of the label Column that is intended to have data generated for
    * @param targetCount The desired number of rows of synthetic data from the class to generate
    * @param fieldsToDrop Fields to be ignored from the DataFrame (that are not part of the feature vector)
    * @tparam T Numeric Type of the Label column
    * @return A DataFrame of the rows that are closest to the K cluster centroids.
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def generateGroupVectors[T: scala.math.Numeric](
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

    val calculatedRowsToGenerate =
      if (rowsToGeneratePerGroup < 1) 1 else rowsToGeneratePerGroup

    centroids
      .map { x =>
        acquireNeighborVectors(
          clusteredData.filter(col(labelCol) === labelGroup),
          lshModel,
          x,
          calculatedRowsToGenerate
        )
      }
      .reduce(_.union(_))
      .drop(fieldsToDrop: _*)
      .limit(targetCount)

  }

  /**
    * Helper Method for converting from Spark DataType to scala types.
    * @param sparkType The spark type of a column
    * @return a string representation of the native scala type
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def sparkToScalaTypeConversion(
    sparkType: DataType
  ): String = {

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

  /**
    * Method for generating the original DataFrame's schema information, storing it in a collection that defines
    * both the full starting schema types as well as information about the feature vector that was passed in.
    * @param fullSchema The full schema as a StructType collection from the input DataFrame
    * @param fieldsNotInVector Array of field names to ignore
    * @return SchemaDefinitions collection payload
    * @author Ben Wilson
    * @since 0.5.1
    */
  private[feature] def generateSchemaInformationPayload(
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
    * Helper method for converting the generated data DataFrame's types to the original types.
    * @note This is critical in order to be able to re-join the data back to the original DataFrame with the correct
    *       types for each field.
    * @param dataFrame The synthetic DataFrame with all fields as DoubleType
    * @param schemaPayload the original DataFrame's type information payload
    * @return a converted types DataFrame
    */
  private[feature] def castColumnsToCorrectTypes(
    dataFrame: DataFrame,
    schemaPayload: SchemaDefinitions
  ): DataFrame = {

    // Extract the fields and types that are part of the feature vector.
    val featureFieldsPayload = schemaPayload.features
      .map(x => x.fieldName)
      .flatMap(
        y => schemaPayload.fullSchema.filter(z => y.contains(z.fieldName))
      )

    // Perform casting by applying the original DataTypes to the feature vector fields.
    featureFieldsPayload
      .foldLeft(dataFrame) {
        case (accum, x) =>
          accum.withColumn(x.fieldName, dataFrame(x.fieldName).cast(x.dfType))
      }

  }

  /**
    * Method for filling in with dummy data any field that was not part of the feature vector
    * @param dataFrame Input DataFrame with feature fields
    * @param schemaPayload schema definitions from original dataframe
    * @param featureCol name of the feature field
    * @param labelCol name of the label field
    * @return the dataframe with any of the missing fields populated with dummy data of the correct type.
    */
  private[feature] def fillMissingColumns(dataFrame: DataFrame,
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
    * Method for rebuilding the feature vector in the same manner as the original DataFrame's feature vector
    * @param dataFrame The Synethic data DataFrame
    * @param featureFields The indexed feature fields for re-creating the original DataFrame's feature vector
    * @return The synthetic DataFrame with an added feature vector column
    */
  private[feature] def rebuildFeatureVector(
    dataFrame: DataFrame,
    featureFields: Array[RowMapping]
  ): DataFrame = {

    val assembler = new VectorAssembler()
      .setInputCols(featureFields.map(_.fieldName))
      .setOutputCol(conf.featuresCol)

    assembler.transform(dataFrame.drop(conf.featuresCol))
  }

  case class MapTypeVal(colName: String, colValue: Column)

  private def addDummyDataForIgnoredColumns(
    dataframe: DataFrame,
    fieldsToIgnore: Array[StructField]
  ): DataFrame = {
    var newDataFrame: DataFrame = dataframe

    val dummyDate = new Date()
    val dummyTime = Calendar.getInstance().getTime

    fieldsToIgnore
      .map(
        item =>
          item.dataType match {
            case StringType    => MapTypeVal(item.name, lit("DUMMY"))
            case IntegerType   => MapTypeVal(item.name, lit(0))
            case DoubleType    => MapTypeVal(item.name, lit(0.0))
            case FloatType     => MapTypeVal(item.name, lit(0.0f))
            case LongType      => MapTypeVal(item.name, lit(0L))
            case ByteType      => MapTypeVal(item.name, lit("DUMMY".getBytes))
            case BooleanType   => MapTypeVal(item.name, lit(false))
            case BinaryType    => MapTypeVal(item.name, lit(0))
            case DateType      => MapTypeVal(item.name, lit(dummyDate))
            case TimestampType => MapTypeVal(item.name, lit(dummyTime))
            case _ =>
              throw new UnsupportedOperationException(
                s"Field '${item.name}' is of type ${item.dataType}, which is not supported."
              )
        }
      )
      .foreach { m: MapTypeVal =>
        newDataFrame = newDataFrame.withColumn(m.colName, m.colValue)
      }

    newDataFrame
  }

  /**
    * Main Method for generating synthetic data
    * @param labelValues Array[RowGenerationConfig] for specifying which categorical labels and the target counts to
    *                    generate data for
    * @return A synthetic data DataFrame with an added field for specifying that this data is synthetic in nature.
    */
  def makeRows(labelValues: Array[RowGenerationConfig]): DataFrame = {

    val collectedFieldsToIgnore = conf.fieldsToIgnore ++ Array(
      conf.featuresCol,
      conf.labelCol
    )

    // Get the schema information
    val ignoredFieldsTypes =
      df.schema.fields.filter(field => conf.fieldsToIgnore.contains(field.name))
    val origSchema = df.schema.names
    val schemaMappings =
      generateSchemaInformationPayload(df.schema, collectedFieldsToIgnore)

    val labelColumnType =
      schemaMappings.fullSchema
        .filter(x => x.fieldName == _labelCol)
        .head
        .dfType

    val doublesSchema =
      generateDoublesSchema(df, collectedFieldsToIgnore.toList)

    // Scale the feature vector
    val scaled = scaleFeatureVector(df)

    // Build a KMeans Model
    val kModel = buildKMeans(scaled)

    // Build a MinHashLSHModel
    val lshModel = buildLSH(scaled)

    // Transform the scaled data with the KMeans model
    val kModelData = kModel.transform(scaled)

    val returnfinalDf = labelValues
      .map { x =>
        val vecs = acquireNearestVectorToCentroids(
          scaled.filter(col(conf.labelCol) === x.labelValue),
          lshModel,
          kModel
        )
        val groupData = generateGroupVectors(
          kModelData,
          vecs,
          lshModel,
          conf.labelCol,
          x.labelValue,
          x.targetCount,
          fieldsToDrop
        )
        val rowCollections = acquireRowCollections(
          groupData.drop(conf.featuresCol),
          x.targetCount,
          conf.minimumVectorCountToMutate,
          conf.mutationMode,
          conf.mutationValue
        )
        val convertedDF =
          convertCollectionsToDataFrame(rowCollections, doublesSchema)
        val finalDF = castColumnsToCorrectTypes(convertedDF, schemaMappings)
        // rebuild the feature vector
        rebuildFeatureVector(finalDF, schemaMappings.features)
          .withColumn(conf.labelCol, lit(x.labelValue))
      }
      .reduce(_.union(_))
      .toDF()

    addDummyDataForIgnoredColumns(returnfinalDf, ignoredFieldsTypes)
      .select(origSchema map col: _*)
      .withColumn(conf.syntheticCol, lit(true))
      .withColumn(_labelCol, col(_labelCol).cast(labelColumnType))

  }

}

object KSampling extends KSamplingBase {

  def apply(data: DataFrame,
            labelValues: Array[RowGenerationConfig],
            featuresCol: String,
            labelsCol: String,
            syntheticCol: String,
            fieldsToIgnore: Array[String],
            kGroups: Int,
            kMeansMaxIter: Int,
            kMeansTolerance: Double,
            kMeansDistanceMeasurement: String,
            kMeansSeed: Long,
            kMeansPredictionCol: String,
            lshHashTables: Int,
            lshSeed: Long,
            lshOutputCol: String,
            quorumCount: Int,
            minimumVectorCountToMutate: Int,
            vectorMutationMethod: String,
            mutationMode: String,
            mutationValue: Double): DataFrame =
    new KSampling(data)
      .setFeaturesCol(featuresCol)
      .setLabelCol(labelsCol)
      .setSyntheticCol(syntheticCol)
      .setFieldsToIgnore(fieldsToIgnore)
      .setKGroups(kGroups)
      .setKMeansMaxIter(kMeansMaxIter)
      .setKMeansTolerance(kMeansTolerance)
      .setKMeansDistanceMeasurement(kMeansDistanceMeasurement)
      .setKMeansSeed(kMeansSeed)
      .setKMeansPredictionCol(kMeansPredictionCol)
      .setLSHHashTables(lshHashTables)
      .setLSHOutputCol(lshOutputCol)
      .setQuorumCount(quorumCount)
      .setMinimumVectorCountToMutate(minimumVectorCountToMutate)
      .setVectorMutationMethod(vectorMutationMethod)
      .setMutationMode(mutationMode)
      .setMutationValue(mutationValue)
      .makeRows(labelValues)

}
