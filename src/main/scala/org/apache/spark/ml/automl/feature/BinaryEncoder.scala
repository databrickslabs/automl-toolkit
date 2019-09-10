package org.apache.spark.ml.automl.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{
  HasHandleInvalid,
  HasInputCols,
  HasOutputCols
}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

trait BinaryEncoderBase
    extends Params
    with HasHandleInvalid
    with HasInputCols
    with HasOutputCols {

  /**
    * Configuration of the Parameter for handling invalid entries in a previously modeled feature column.
    */
  override val handleInvalid: Param[String] = new Param[String](
    this,
    "handleInvalid",
    "Handling invalid data flag for utilizing the BinaryEncoderModel during transform method call." +
      "Options are: 'keep' (encodes unknown values as Binary 0's as an unknown categorical class) or " +
      "'error' (throw error if unknown value is introduced).",
    ParamValidators.inArray(BinaryEncoder.supportedHandleInvalids)
  )

  setDefault(handleInvalid, BinaryEncoder.ERROR_INVALID)

  /**
    * Method for validating the resultant schema from the application of building and transforming using this
    * encoder package.  The purpose of validation is to ensure that the supplied input columns are of the correct
    * binary or nominal (ordinal numeric) type and that the output columns will contain the correct number of columns
    * based on the configuration set.
    * @param schema The schema of the dataset supplied for training of the model or used in transforming using the model
    * @param keepInvalid Boolean flag for whether to allow for an additional binary encoding value to be used for
    *                    any values that were unknown at the time of model training, which will summarily be
    *                    converted to a 'max binary value' of the encoding length + 1 with maximum n * "1" values.
    * @return StructType that represents the transformed schema with additional output columns appended to the
    *         dataset structure.
    * @since 0.5.3
    * @throws UnsupportedOperationException if the configured input cols and output cols do not match one another in
    *                                       length.
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[UnsupportedOperationException])
  protected def validateAndTransformSchema(schema: StructType,
                                           keepInvalid: Boolean): StructType = {

    val inputColNames = $(inputCols)
    val outputColNames = $(outputCols)

    require(
      inputColNames.length == outputColNames.length,
      s"The supplied number of input columns " +
        s"${inputColNames.length} to BinaryEncoder" +
        s"do not match the output columns count ${outputColNames.length}.\n InputCols: ${inputColNames
          .mkString(", ")}" +
        s"\n OutputCols: ${outputColNames.mkString(", ")}"
    )

    // Validate that the supplied input columns are of numeric type
    inputColNames.foreach(SchemaUtils.checkNumericType(schema, _))

    val inputFields = $(inputCols).map(schema(_))

    val outputFields = inputFields.zip(outputColNames).map {
      case (inputField, outputColName) =>
        BinaryEncoderCommon
          .transformOutputColumnSchema(inputField, outputColName, keepInvalid)
    }
    outputFields.foldLeft(schema) {
      case (newSchema, outputField) =>
        StructType(newSchema.fields :+ outputField)
    }
  }

}

class BinaryEncoder(override val uid: String)
    extends Estimator[BinaryEncoderModel]
    with DefaultParamsWritable
    with HasInputCols
    with HasOutputCols
    with BinaryEncoderBase {

  def this() = this(Identifiable.randomUID("binaryEncoder"))

  /**
    * Setter for supplying the array of input columns to be encoded with the BinaryEncoder type
    * @param values Array of column names
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  def setInputCols(values: Array[String]): this.type = set(inputCols, values)

  /**
    * Setter for supplying the array of output columns that are the result of running a .transform from a trained
    * model on an appropriate dataset of compatible schema
    * @param values Array of column names that will be generated through a .transform
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  def setOutputCols(values: Array[String]): this.type = set(outputCols, values)

  /**
    * Setter for supplying an optional 'keep' or 'error' (Default: 'error') for un-seen values that arrive into a
    * pre-trained model.  With the 'keep' setting, an additional vector position is added to the output column
    * to ensure no collisions may exist with real data and the values throughout each of the Array[Double] locations
    * in the DenseVector output will all be set to '1'
    * @param value String: either 'keep' or 'error' (Default: 'error')
    * @throws SparkException if the configuration value supplied is not either 'keep' or 'error'
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[SparkException])
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

  override def transformSchema(schema: StructType): StructType = {
    val keepInvalid = $(handleInvalid) == BinaryEncoder.KEEP_INVALID
    validateAndTransformSchema(schema, keepInvalid = keepInvalid)
  }

  /**
    * Main fit method that will build a BinaryEncoder model from the data set and the configured input and output columns
    * specified in the setters.
    * The primary principle at work here is dimensionality reduction for the encoding of extremely high-cardinality
    * StringIndexed columns.  OneHotEncoding works extremely well for this purpose, but has the side-effect of
    * requiring extremely large amounts of columns to be generated when performing OHE is increased memory pressure.
    * This package allows for a lossy reduction in this space by distilling the information into a binary string
    * encoding space that is dynamic based on the encoded length of the maximum nominal space as represented in binary
    *
    * @example e.g. if the cardinality of a nominal column is 113, the binary representation of that is 1110001.
    *          When using OHE, this would result in 113 (or 114 if allowing invalids) binary positions within a sparse
    *          vector, creating 113 or 114 columns in the dataset.  However, using BinaryEncoder, we are left with 7
    *          (or 8, if allowing invalids) dense vector positions to capture the same amount of information.
    *
    * @note Due to the nature of this encoding and how the majority of models learn, this is seen as an information
    *       loss encoding.  However, considering that high cardinality non-numeric nominal fields are frequently
    *       discarded due to the explosion of the data set, this is providing the ability to utilize high cardinality
    *       fields that otherwise would not be able to be included.
    * @param dataset The dataset (or DataFrame) used in training the model
    * @return BinaryEncoderModel - a serializable artifact that has the output schema and encoding embedded within it.
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  override def fit(dataset: Dataset[_]): BinaryEncoderModel = {

    transformSchema(dataset.schema)

    val mutatedSchema =
      validateAndTransformSchema(dataset.schema, keepInvalid = false)

    // Build an array of the size of the targeted binary arrays to be encoded
    val categoricalColumnSizes = new Array[Int]($(outputCols).length)

    // populate the Array of categorical column sizes with the size information for fitting of the model
    val columnToScanIndices = $(outputCols).zipWithIndex.flatMap {
      case (outputColName, idx) =>
        val classCount =
          AttributeGroup.fromStructField(mutatedSchema(outputColName)).size
        if (classCount < 0) {
          Some(idx)
        } else {
          categoricalColumnSizes(idx) = classCount
          None
        }
    }

    // If the metadata doesn't have the attribute information, extract it manually.
    if (columnToScanIndices.length > 0) {
      val inputColNames = columnToScanIndices.map($(inputCols)(_))
      val outputColNames = columnToScanIndices.map($(outputCols)(_))

      val attrGroups = BinaryEncoderCommon.getOutputAttrGroupFromData(
        dataset,
        inputColNames,
        outputColNames
      )
      attrGroups.zip(columnToScanIndices).foreach {
        case (attrGroup, idx) =>
          categoricalColumnSizes(idx) = attrGroup.size
      }
    }

    val model =
      new BinaryEncoderModel(uid, categoricalColumnSizes).setParent(this)
    copyValues(model)

  }

  override def copy(extra: ParamMap): BinaryEncoder = defaultCopy(extra)

}

object BinaryEncoder extends DefaultParamsReadable[BinaryEncoder] {

  private[feature] val KEEP_INVALID: String = "keep"
  private[feature] val ERROR_INVALID: String = "error"
  private[feature] val supportedHandleInvalids: Array[String] =
    Array(KEEP_INVALID, ERROR_INVALID)

  override def load(path: String): BinaryEncoder = super.load(path)

}

class BinaryEncoderModel(override val uid: String,
                         val categorySizes: Array[Int])
    extends Model[BinaryEncoderModel]
    with BinaryEncoderBase
    with MLWritable {

  import BinaryEncoderModel._

  /**
    * Helper method for adjusting the sizes that the Binary Encoder will need to encode to capture the BinaryString
    * representation of the Nominal data, adjusted for the keepInvalid flag.
    * @since 0.5.3
    */
  private def getConfigedCategorySizes: Array[Int] = {

    val keepInvalid = getHandleInvalid == BinaryEncoder.KEEP_INVALID

    if (keepInvalid) {
      categorySizes.map(_ + 1)
    } else {
      categorySizes
    }

  }

  /**
    * Main UDF for performing the conversion of nominal / binary encoded data from StringIndexer to a BinaryString format
    * as a Breeze DenseVector
    * @throws SparkException if the value for encoding is not within the column index during training
    *                        (keepInvalid set to 'error')
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[SparkException])
  private def encoder: UserDefinedFunction = {

    val keepInvalid = getHandleInvalid == BinaryEncoder.KEEP_INVALID

    val localCategorySizes = categorySizes

    udf { (label: Any, colIdx: Int) =>
      val origCategorySize = localCategorySizes(colIdx)

      val maxCategorySize =
        BinaryEncoderCommon.convertToBinaryString(Some(origCategorySize)).length

      // Add additional index position to vector if the
      val idxLength = if (keepInvalid) maxCategorySize + 1 else maxCategorySize

      val encodedData =
        BinaryEncoderCommon.convertToBinary(Some(label), idxLength)

      val idx = if (encodedData.length <= origCategorySize) {
        encodedData
      } else {
        if (keepInvalid) {
          Array.fill(maxCategorySize)(1.0)
        } else {
          throw new SparkException(
            s"The value specified for Binary Encoding (${label.toString}) in column index " +
              s"$colIdx has not been seen" +
              s"during training of this model.  To enable reclassification of unseen values, set the handleInvalid" +
              s"parameter to ${BinaryEncoder.KEEP_INVALID}"
          )
        }
      }
      Vectors.dense(idx)
    }
  }

  /**
    * Setter for specifying the column names in Array format for the columns intended to be Binary Indexed.
    * @param values Array of column names
    * @since 0.5.3
    * @author Ben Wilson, DataBricks
    */
  def setInputCols(values: Array[String]): this.type = set(inputCols, values)

  /**
    * Setter for specifying the desired output columns in Array format for the columns to be generated as Breeze
    * DenseVectors when the model is used to transform a dataset
    * @param values Array of output column names
    * @note the index position relationship between setInputCols and setOutputCols is a 1 to 1 relationship.  The
    *       positional order and length must be congruent and match.
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  def setOutputCols(values: Array[String]): this.type = set(outputCols, values)

  /**
    * Setter for whether to allow for unseen indexed nominal values to be used in the transformation of a dataset with
    * the generated BinaryEncoderModel.
    * @note Default: 'error' optional settings: 'keep' or 'error'
    * @param value The setting to be used: either 'keep' or 'error'
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

  /**
    * Method for mutating the dataset schema to support the addition of BinaryEncoded columns
    * @param schema the schema of the dataset
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  override def transformSchema(schema: StructType): StructType = {

    val inputColNames = $(inputCols)

    require(
      inputColNames.length == categorySizes.length,
      s"The number of input columns specified " +
        s"(${inputColNames.length}) must be the same number of feature columns during the fit (${categorySizes.length})"
    )

    val keepInvalid = $(handleInvalid) == BinaryEncoder.KEEP_INVALID

    val transformedSchema =
      validateAndTransformSchema(schema, keepInvalid = keepInvalid)
    verifyNumOfValues(transformedSchema)

  }

  /**
    * Private method for validating that the schema metadata matches the expected cardinality for the column to be encoded
    * @param schema schema of the input dataset
    * @throws IllegalArgumentException if the metadata information does not match the data cardinality
    * @return the schema of the dataset (pass-through method)
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[IllegalArgumentException])
  private def verifyNumOfValues(schema: StructType): StructType = {
    val configedSizes = getConfigedCategorySizes
    $(outputCols).zipWithIndex.foreach {
      case (outputColName, idx) =>
        val inputColName = $(inputCols)(idx)
        val attrGroup = AttributeGroup.fromStructField(schema(outputColName))

        if (attrGroup.attributes.nonEmpty) {
          val numCategories = configedSizes(idx)
          require(
            attrGroup.size == numCategories,
            s"The number of distinct values in column $inputColName" +
              s"was expected to be $numCategories, but the metadata shows ${attrGroup.size} distinct values."
          )
        }
    }
    schema
  }

  /**
    * Main transformation method that will apply the model's configured encoding through a udf to the input dataset
    * and add encoded columns.
    * @param dataset input dataset for the model to mutate
    * @return a DataFrame with added BinaryEncoded columns
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  override def transform(dataset: Dataset[_]): DataFrame = {

    val transformedSchema = transformSchema(dataset.schema, logging = true)

    val keepInvalid = $(handleInvalid) == BinaryEncoder.KEEP_INVALID

    val encodedColumns = $(inputCols).indices.map { idx =>
      val inputColName = $(inputCols)(idx)
      val outputColName = $(outputCols)(idx)

      val metadata = BinaryEncoderCommon
        .createAttrGroupForAttrNames(
          outputColName,
          categorySizes(idx),
          keepInvalid
        )
        .toMetadata()

      encoder(col(inputColName).cast(DoubleType), lit(idx))
        .as(outputColName, metadata)
    }

    dataset.withColumns($(outputCols), encodedColumns)

  }

  override def copy(extra: ParamMap): BinaryEncoderModel = {
    val copied = new BinaryEncoderModel(uid, categorySizes)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new BinaryEncoderModelWriter(this)

}

object BinaryEncoderModel extends MLReadable[BinaryEncoderModel] {
  private[BinaryEncoderModel] class BinaryEncoderModelWriter(
    instance: BinaryEncoderModel
  ) extends MLWriter {

    private case class Data(categorySizes: Array[Int])

    override protected def saveImpl(path: String): Unit = {

      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.categorySizes)
      val dataPath = new Path(path, "data").toString
      sparkSession
        .createDataFrame(Seq(data))
        .repartition(1)
        .write
        .parquet(dataPath)
    }
  }

  private class BinaryEncoderModelReader extends MLReader[BinaryEncoderModel] {

    private val className = classOf[BinaryEncoderModel].getName

    override def load(path: String): BinaryEncoderModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read
        .parquet(dataPath)
        .select("categorySizes")
        .head()
      val categorySizes = data.getAs[Seq[Int]](0).toArray
      val model = new BinaryEncoderModel(metadata.uid, categorySizes)
      metadata.getAndSetParams(model)
      model
    }

  }

  override def read: MLReader[BinaryEncoderModel] = new BinaryEncoderModelReader

  override def load(path: String): BinaryEncoderModel = super.load(path)

}

private[feature] object BinaryEncoderCommon {

  /**
    * Helper method for appropriately padding leading zero's based on the BinaryString Length value and the
    * target encoding length
    * @param inputString A supplied BinaryString encoded value
    * @throws IllegalArgumentException if the target encoding size is smaller than the data size after encoding,
    *                                  since this would result in significant data loss and issues with modeling.
    * @return Padded string to the prescribed encoding length
    */
  @throws(classOf[IllegalArgumentException])
  private[feature] def padZeros(inputString: String,
                                encodingSize: Int): String = {

    val deltaLength = encodingSize - inputString.length

    deltaLength match {
      case 0          => inputString
      case x if x > 0 => "0" * x + inputString
      case _ =>
        throw new IllegalArgumentException(
          s"Target encoding size $encodingSize of BinaryString is less " +
            s"than total encoded information.  Information loss of a substantial degree would be generated. " +
            s"Adjust encodingSize higher."
        )
    }

  }

  /**
    * Private encoding method.
    * @param value takes in an Option of Any type and casts it to a BinaryString representation
    * @note Conversions from non-whole-number values of numeric types WILL incur information loss as decimal values
    *       cannot be represented by simple BinaryString.  It will incur a rounding to the nearest whole number.
    * @tparam A Any type
    * @return Binary String conversion of the input data
    * @throws UnsupportedOperationException if the data type being input is not of the correct type for conversion.
    */
  @throws(classOf[UnsupportedOperationException])
  private[feature] def convertToBinaryString[A <: Any](
    value: Option[A]
  ): String = {

    value.get match {
      case a: Boolean => if (a) "1" else "0"
      case a: Byte    => a.toByte.toBinaryString
      case a: Char    => a.toChar.toBinaryString
      case a: Int     => a.toInt.toBinaryString
      case a: Long    => a.toLong.toBinaryString
      case a: Float   => a.toFloat.toByte.toBinaryString
      case a: Double  => a.toString.toDouble.toByte.toBinaryString
      case a: String =>
        a.toString.toCharArray
          .flatMap(_.toBinaryString)
          .mkString("")
      case a: BigDecimal => a.toString.toByte.toBinaryString
      case _ =>
        throw new UnsupportedOperationException(
          s"ordinalToBinary does not support type :" +
            s"${value.getClass.getSimpleName}"
        )
    }

  }

  /**
    * Private method for converting Any type to and Array of Doubles with appropriate prefix padding to ensure
    * constant vector length of the output DenseVector from transformation.
    * @param ordinalValue The value to encode
    * @param encodingSize The size of the Array of Binary Double values for output
    * @tparam A Any type (primitives are only supported - collections will throw an exception)
    * @return Array of Doubles, representing the Binary values of the encoded value
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  private[feature] def convertToBinary[A <: Any](
    ordinalValue: Option[A],
    encodingSize: Int
  ): Array[Double] = {

    val binaryString = convertToBinaryString(ordinalValue)
    val padded = padZeros(binaryString, encodingSize)
    binaryStringToDoubleArray(padded)
  }

  /**
    * Private method for converting a binary string to Array[Double]
    * @param binary Binary Encoded String
    * @return Array[Double]  for the numeric vector-representation of an encoded value in Binary format
    * @since 0.5.3
    * @author Ben Wilson, Databricks
    */
  private[feature] def binaryStringToDoubleArray(
    binary: String
  ): Array[Double] = {

    binary.toCharArray.map(_.toString.toDouble)

  }

  /**
    * Pulled from OneHotEncoder with slight modifications.
    * @param dataset Input dataset
    * @param inputColNames The names that are to be encoded
    * @param outputColNames the output columns after encoding
    * @return Seq[AttributeGroup] for the schema metadata associated with the Model
    * @since 0.5.3
    */
  def getOutputAttrGroupFromData(
    dataset: Dataset[_],
    inputColNames: Seq[String],
    outputColNames: Seq[String]
  ): Seq[AttributeGroup] = {

    val columns = inputColNames.map { inputColName =>
      col(inputColName).cast(DoubleType)
    }
    val numOfColumns = columns.length

    val numAttrsArray = dataset
      .select(columns: _*)
      .rdd
      .map { row =>
        (0 until numOfColumns).map(idx => row.getDouble(idx)).toArray
      }
      .treeAggregate(new Array[Double](numOfColumns))(
        (maxValues: Array[Double], curValues: Array[Double]) => {
          (0 until numOfColumns).foreach {
            idx =>
              val x = curValues(idx)
              assert(
                x <= Int.MaxValue,
                s"Index out of range for maximum ordinal Value for 32bit int space.  " +
                  s"Value: $x is larger than maximum of ${Int.MaxValue} in field ${inputColNames(idx)}"
              )
              assert(
                x >= 0.0 && x == x.toInt,
                s"Values from column ${inputColNames(idx)} must be indices, but got value $x, which is invalid."
              )
              maxValues(idx) = math.max(maxValues(idx), x)
          }
          maxValues
        },
        (m0, m1) => {
          (0 until numOfColumns).foreach { idx =>
            m0(idx) = math.max(m0(idx), m1(idx))
          }
          m0
        }
      )
      .map(_.toInt + 1)

    outputColNames.zip(numAttrsArray).map {
      case (outputColName, numAttrs) =>
        createAttrGroupForAttrNames(
          outputColName,
          numAttrs,
          keepInvalid = false
        )
    }

  }

  /**
    * Pulled from OneHotEncoder with slight modifications.
    * @param outputColName Output Column to be created by the BinaryEncoderModel
    * @param numAttrs The number of unique values for the output column
    * @param keepInvalid Boolean flag for whether to allow for unseen values to be grouped into an unknown binary
    *                    representation during transformation
    * @return AttributeGroup
    * @since 0.5.3
    */
  def createAttrGroupForAttrNames(outputColName: String,
                                  numAttrs: Int,
                                  keepInvalid: Boolean): AttributeGroup = {

    val maxAttributeSize = if (keepInvalid) {
      BinaryEncoderCommon.convertToBinaryString(Some(numAttrs)).length + 1
    } else {
      BinaryEncoderCommon.convertToBinaryString(Some(numAttrs)).length
    }

    val outputAttrNames = Array.tabulate(maxAttributeSize)(_.toString)

    genOutputAttrGroup(Some(outputAttrNames), outputColName)
  }

  /**
    * Pulled from OneHotEncoder with slight modifications.
    * @param outputAttrNames Attributes for the metadata
    * @param outputColName The name of the output column to be added
    * @return AttributeGroup
    * @since 0.5.3
    */
  private def genOutputAttrGroup(outputAttrNames: Option[Array[String]],
                                 outputColName: String): AttributeGroup = {
    outputAttrNames
      .map { attrNames =>
        val attrs: Array[Attribute] = attrNames.map { name =>
          BinaryAttribute.defaultAttr.withName(name)
        }
        new AttributeGroup(outputColName, attrs)
      }
      .getOrElse {
        new AttributeGroup(outputColName)
      }
  }

  /**
    * Pulled from OneHotEncoder with slight modifications.
    * @param inputCol An input column to extract the encoding lengths for the output column during transformation
    * @return Output names for attritubtes
    */
  private def genOutputAttrNames(
    inputCol: StructField
  ): Option[Array[String]] = {
    val inputAttr = Attribute.fromStructField(inputCol)

    inputAttr match {
      case nominal: NominalAttribute =>
        val outputCardinality =
          BinaryEncoderCommon
            .convertToBinaryString(Some(nominal.values.get.length))
            .length
        Some((0 to outputCardinality).toArray.map(_.toString))
      case binary: BinaryAttribute =>
        if (binary.values.isDefined) {
          binary.values
        } else {
          Some(Array.tabulate(2)(_.toString))
        }
      case _: NumericAttribute =>
        throw new RuntimeException(
          s"The input column ${inputCol.name} cannot be continuous-value."
        )
      case _ =>
        None // optimistic about unknown attributes
    }
  }

  /**
    * Pulled from OneHotEncoder with slight modifications.
    * @param inputCol An input column that will be used to validate the output column corresponding to it.
    * @param outputColName An output column to match to the input column
    * @param keepInvalid invalid retention flag
    * @return the Output column struct field definition
    */
  def transformOutputColumnSchema(inputCol: StructField,
                                  outputColName: String,
                                  keepInvalid: Boolean = false): StructField = {

    val outputAttrNames = genOutputAttrNames(inputCol)
    val filteredOutputAttrNames = outputAttrNames.map { names =>
      if (keepInvalid) {
        names ++ Seq("invalidValues")
      } else {
        names
      }
    }
    genOutputAttrGroup(filteredOutputAttrNames, outputColName).toStructField()
  }

}
