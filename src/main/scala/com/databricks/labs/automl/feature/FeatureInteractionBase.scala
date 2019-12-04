package com.databricks.labs.automl.feature

import com.databricks.labs.automl.exceptions.ModelingTypeException
import com.databricks.labs.automl.feature.structures._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

trait FeatureInteractionBase {
  import com.databricks.labs.automl.feature.structures.FieldEncodingType._
  import com.databricks.labs.automl.feature.structures.InteractionRetentionMode._
  import com.databricks.labs.automl.feature.structures.ModelingType._

  private final val allowableModelTypes = Array("classifier", "regressor")
  private final val allowableFieldTypes = Array("nominal", "continuous")
  private final val allowableRetentionModes =
    Array("optimistic", "strict", "all")

  final val AGGREGATE_COLUMN: String = "totalCount"
  final val COUNT_COLUMN: String = "count"
  final val RATIO_COLUMN: String = "labelRatio"
  final val TOTAL_RATIO_COLUMN: String = "totalRatio"
  final val ENTROPY_COLUMN: String = "entropy"
  final val FIELD_ENTROPY_COLUMN: String = "fieldEntropy"
  final val QUANTILE_THRESHOLD: Double = 0.5
  final val QUANTILE_PRECISION: Double = 0.95
  final val VARIANCE_STATISTIC: String = "stddev"
  final val INDEXED_SUFFIX: String = "_si"

  protected[feature] def getModelType(
    modelingType: String
  ): ModelingType.Value = {
    modelingType match {
      case "regressor"  => Regressor
      case "classifier" => Classifier
      case _            => throw ModelingTypeException(modelingType, allowableModelTypes)
    }
  }

  protected[feature] def getFieldType(
    fieldType: String
  ): FieldEncodingType.Value = {
    fieldType match {
      case "nominal"    => Nominal
      case "continuous" => Continuous
      case _            => throw ModelingTypeException(fieldType, allowableFieldTypes)
    }
  }

  protected[feature] def getRetentionMode(
    retentionMode: String
  ): InteractionRetentionMode.Value = {
    retentionMode match {
      case "optimistic" => Optimistic
      case "strict"     => Strict
      case "all"        => All
      case _ =>
        throw ModelingTypeException(retentionMode, allowableRetentionModes)
    }
  }

  /**
    * Method for generating a collection of Interaction Candidates to be tested and applied to the feature set
    * if the tests for inclusion pass.
    * @param featureColumns List of the columns that make up the feature vector
    * @return Array of InteractionPayload values.
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  protected[feature] def generateInteractionCandidates(
    featureColumns: Array[ColumnTypeData]
  ): Array[InteractionPayload] = {
    val colIdx = featureColumns.zipWithIndex
    colIdx.flatMap {
      case (x, i) =>
        val maxIdx = colIdx.length
        for (j <- Range(i + 1, maxIdx)) yield {
          InteractionPayload(
            x.name,
            x.dataType,
            colIdx(j)._1.name,
            colIdx(j)._1.dataType,
            s"i_${x.name}_${colIdx(j)._1.name}"
          )
        }
    }
  }

  /**
    * Method for evaluating the percentage change to the score metric to normalize.
    * @param before Score of a parent feature
    * @param after Score of an interaction feature
    * @return the percentage change
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  protected[feature] def calculatePercentageChange(before: Double,
                                                   after: Double): Double = {

    (after - before) / math.abs(before) * 100.0

  }

  /**
    * Method for generating a product interaction between feature columns
    * @param df A DataFrame to add a field for an interaction between two columns
    * @param candidate InteractionPayload information about the two parent columns and the name of the new interaction column to be created.
    * @return A modified DataFrame with the new column.
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  protected[feature] def interactProduct(
    df: DataFrame,
    candidate: InteractionPayload
  ): DataFrame = {

    df.withColumn(
      candidate.outputName,
      col(candidate.left) * col(candidate.right)
    )

  }

  /**
    * Method for converting nominal interaction fields to a new StringIndexed value to preserve information type and
    * eliminate the possibility of data distribution skew
    * @param payload FeatureInteractionCollection of the source parents and their interacted children fields
    * @return NominalDataCollecction payload containing a DataFrame that has new StringIndexed fields for nominal
    *         interactions and the fields that need to be seen as included in the final feature vector
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  protected[feature] def generateNominalIndexesInteractionFields(
    payload: FeatureInteractionCollection
  ): NominalDataCollection = {

    // Check for nominal data types on interactions

    val parsedNames = payload.interactionPayload
      .map(
        x =>
          (x.rightDataType, x.leftDataType) match {
            case ("nominal", "nominal") =>
              NominalIndexCollection(x.outputName, indexCheck = true)
            case _ => NominalIndexCollection(x.outputName, indexCheck = false)
        }
      )

    val nominalFields = parsedNames
      .filter(x => x.indexCheck)
      .map(x => x.name)

    // String Index these fields

    val indexers = nominalFields.map { x =>
      new StringIndexer()
        .setHandleInvalid("keep")
        .setInputCol(x)
        .setOutputCol(x + INDEXED_SUFFIX)
    }

    val pipeline = new Pipeline()
      .setStages(indexers)
      .fit(payload.data)

    val adjustedFieldsToIncludeInVector = parsedNames.map { x =>
      if (x.indexCheck) x.name + INDEXED_SUFFIX
      else x.name
    }

    NominalDataCollection(
      pipeline.transform(payload.data),
      adjustedFieldsToIncludeInVector,
      nominalFields,
      indexers
    )

  }

  /**
    * Helper method for recreating the feature vector after interactions have been completed on individual columns
    * @param df DataFrame containing the interacted fields with the original feature vector dropped
    * @param preInteractedFields Fields making up the original vector before interaction
    * @param interactedFields Interaction candidate fields that have been selected to be included in the final feature vector
    * @param featureCol Name of the feature vector field
    * @return DataFrame with a new feature vector.
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  protected[feature] def regenerateFeatureVector(
    df: DataFrame,
    preInteractedFields: Array[String],
    interactedFields: Array[String],
    featureCol: String
  ): VectorAssemblyOutput = {

    val assembler = new VectorAssembler()
      .setInputCols(preInteractedFields ++ interactedFields)
      .setOutputCol(featureCol)

    VectorAssemblyOutput(assembler, assembler.transform(df))

  }

}
