package org.apache.spark.ml.automl.feature

import com.databricks.labs.automl.utils.SparkSessionWrapper
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasHandleInvalid, HasInputCols, HasOutputCols}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DoubleType, NumericType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.{SparkContext, SparkException}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{JObject, _}

trait BinaryEncoderBase
    extends Params
    with HasHandleInvalid
    with HasInputCols
    with HasOutputCols {

  override val handleInvalid: Param[String] = new Param[String](
    this,
    "handleInvalid",
    "Handling invalid data flag for utilizing the BinaryEncoderModel during transform method call." +
      "Options are: 'keep' (encodes unknown values as Binary 0's as an unknown categorical class) or " +
      "'error' (throw error if unknown value is introduced).",
    ParamValidators.inArray(BinaryEncoder.supportedHandleInvalids)
  )

  setDefault(handleInvalid, BinaryEncoder.ERROR_INVALID)

  // This has to be defined here since NumericType validation is hidden behind the protection of [spark]
  private[encoders] def checkNumericType(schema: StructType,
                                         colName: String): Unit = {

    val columnDataType = schema(colName).dataType

    require(
      columnDataType.isInstanceOf[NumericType],
      s"Column $colName must be of numeric type but was actually of type " +
        s"${columnDataType.catalogString}"
    )

  }

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
    inputColNames.foreach(checkNumericType(schema, _))

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

  def setInputCols(values: Array[String]): this.type = set(inputCols, values)

  def setOutputCols(values: Array[String]): this.type = set(outputCols, values)

  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

  override def transformSchema(schema: StructType): StructType = {
    val keepInvalid = $(handleInvalid) == BinaryEncoder.KEEP_INVALID
    validateAndTransformSchema(schema, keepInvalid = keepInvalid)
  }

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
//  private[encoders] def calculateEncodingLength()

}

object BinaryEncoder extends DefaultParamsReadable[BinaryEncoder] {

  private[encoders] val KEEP_INVALID: String = "keep"
  private[encoders] val ERROR_INVALID: String = "error"
  private[encoders] val supportedHandleInvalids: Array[String] =
    Array(KEEP_INVALID, ERROR_INVALID)

  override def load(path: String): BinaryEncoder = super.load(path)

}

class BinaryEncoderModel(override val uid: String,
                         val categorySizes: Array[Int])
    extends Model[BinaryEncoderModel]
    with BinaryEncoderBase
    with MLWritable {

  private def getConfigedCategorySizes: Array[Int] = {

    val keepInvalid = getHandleInvalid == BinaryEncoder.KEEP_INVALID

    if (keepInvalid) {
      categorySizes.map(_ + 1)
    } else {
      categorySizes
    }

  }

  private def encoder: UserDefinedFunction = {

    val keepInvalid = getHandleInvalid == BinaryEncoder.KEEP_INVALID

    val localCategorySizes = categorySizes

    udf { (label: Any, colIdx: Int) =>
      val origCategorySize = localCategorySizes(colIdx)

      val maxCategorySize =
        BinaryEncoderCommon.convertToBinaryString(origCategorySize).length

      // Add additional index position to vector if the
      val idxLength = if (keepInvalid) maxCategorySize + 1 else maxCategorySize

      val encodedData = BinaryEncoderCommon.convertToBinary(label, idxLength)

      val idx = if (encodedData.length < origCategorySize) {
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

  def setInputCols(values: Array[String]): this.type = set(inputCols, values)
  def setOutputCols(values: Array[String]): this.type = set(outputCols, values)
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

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

  private def verifyNumOfValues(schema: StructType): StructType = {
    val configedSizes = getConfigedCategorySizes
    $(outputCols).zipWithIndex.foreach {
      case (outputColName, idx) =>
        val inputColName = $(inputColName)(idx)
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

  override def transform(dataset: Dataset[_]): DataFrame = {

    val transformedSchema = transformSchema(dataset.schema, logging = true)
    //todo finish
  }

}

object BinaryEncoderModel
    extends MLReadable[BinaryEncoderModel] {

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


  }



}

private[encoders] object BinaryEncoderCommon {

  /**
    * Helper method for appropriately padding leading zero's based on the BinaryString Length value and the
    * target encoding length
    * @param inputString A supplied BinaryString encoded value
    * @return Padded string to the prescribed encoding length
    */
  private[encoders] def padZeros(inputString: String,
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

  //TODO: this should all be the output of StringIndexer, so maybe simplify this to just do Double/Int conversion?
  private[encoders] def convertToBinaryString[A <: Any](value: A): String = {

    value match {
      case a: Boolean => if (a) "1" else "0"
      case a: Byte    => a.toByte.toBinaryString
      case a: Char    => a.toChar.toBinaryString //TODO: maybe remove this?
      case a: Int     => a.toInt.toBinaryString
      case a: Long    => a.toLong.toBinaryString
      case a: Float   => a.toFloat.toByte.toBinaryString
      case a: Double  => a.toString.toDouble.toByte.toBinaryString
      case a: String =>
        a.toString.toCharArray
          .flatMap(_.toBinaryString)
          .mkString("") //TODO: maybe remove this?
      case a: BigDecimal => a.toByte.toBinaryString
      case _ =>
        throw new UnsupportedOperationException(
          s"ordinalToBinary does not support type :" +
            s"${value.getClass.getSimpleName}"
        )
    }

  }

  private[encoders] def convertToBinary[A <: Any](
    ordinalValue: A,
    encodingSize: Int
  ): Array[Double] = {

    val binaryString = convertToBinaryString(ordinalValue)
    val padded = padZeros(binaryString, encodingSize)
    binaryStringToDoubleArray(padded)
  }

  private[encoders] def binaryStringToDoubleArray(
    binary: String
  ): Array[Double] = {

    binary.toCharArray.map(_.toString.toDouble)

  }

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

  def createAttrGroupForAttrNames(outputColName: String,
                                  numAttrs: Int,
                                  keepInvalid: Boolean): AttributeGroup = {

    val maxAttributeSize = if (keepInvalid) {
      BinaryEncoderCommon.convertToBinaryString(numAttrs).length + 1
    } else {
      BinaryEncoderCommon.convertToBinaryString(numAttrs).length
    }

    val outputAttrNames = Array.tabulate(maxAttributeSize)(_.toString)

    genOutputAttrGroup(Some(outputAttrNames), outputColName)
  }

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

  private def genOutputAttrNames(
    inputCol: StructField
  ): Option[Array[String]] = {
    val inputAttr = Attribute.fromStructField(inputCol)

    inputAttr match {
      case nominal: NominalAttribute =>
        val outputCardinality =
          BinaryEncoderCommon.convertToBinaryString(nominal.values).length
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

  def transformOutputColumnSchema(inputCol: StructField,
                                  outputColName: String,
                                  keepInvalid: Boolean = false): StructField = {

    val outputAttrNames = genOutputAttrNames(inputCol)

  }

}
