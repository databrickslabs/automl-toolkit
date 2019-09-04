package com.databricks.labs.automl.utils.encoders

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.ml.param.shared.{
  HasHandleInvalid,
  HasInputCols,
  HasOutputCols
}
import org.apache.spark.ml.param.{
  IntParam,
  Param,
  ParamMap,
  ParamValidators,
  Params,
  StringArrayParam
}
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsWritable,
  Identifiable,
  MLWritable,
  SchemaUtils
}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.{NumericType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.udf

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

    val model = new BinaryEncoderModel(uid, categorySizes).setParent(this)
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

  import BinaryEncoderModel._

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
      val encodedLabel = BinaryEncoderCommon.convertToBinaryString(label)
      val maxCategorySize = BinaryEncoderCommon.convertToBinaryString(origCategorySize).length
      // Add additional index position to vector if the 
      val idxLength = if (keepInvalid) maxCategorySize + 1 else maxCategorySize



      val idx =

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

  private[encoders] def ordinalToBinary[A <: Any](ordinalValue: A,
                                                  encodingSize: Int): String = {

    val binaryString = convertToBinaryString(ordinalValue)
    padZeros(binaryString, encodingSize)
  }

}
