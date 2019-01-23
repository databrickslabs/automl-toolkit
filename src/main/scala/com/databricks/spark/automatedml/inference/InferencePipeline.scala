package com.databricks.spark.automatedml.inference

import com.databricks.spark.automatedml.executor.AutomationConfig
import com.databricks.spark.automatedml.sanitize.{DataSanitizer, VarianceFiltering}
import com.databricks.spark.automatedml.utils.{AutomationTools, DataValidation}
import org.apache.spark.sql.DataFrame

class InferencePipeline(df: DataFrame) extends AutomationConfig with AutomationTools with DataValidation{



  // TODO: casting of field types
//  private def castDataTypesConstructor(data: DataFrame, codeGen: String) = {
//
//    val (numericFields, characterFields, dateFields, timeFields) = extractTypes(df, _labelCol, _fieldsToIgnoreInVector)
//
//
//
//  }



  /**
    * Method for determining what na.fill should be for the raw data.
    * @param data raw DataFrame that was used to start the autoML process.
    * @return (String, String) Code Gen String, Model Type(classifier or regressor)
    */
  private def naFillConstructor(data: DataFrame, codeGen: String): NAFillConstructorReturn = {

    val naObject = new DataSanitizer(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .setModelSelectionDistinctThreshold(_mainConfig.fillConfig.modelSelectionDistinctThreshold)
      .setNumericFillStat(_mainConfig.fillConfig.numericFillStat)
      .setCharacterFillStat(_mainConfig.fillConfig.characterFillStat)
      .setParallelism(_mainConfig.geneticConfig.parallelism)
      .setFieldsToIgnoreInVector(_mainConfig.fieldsToIgnoreInVector)

    val modelType = naObject.decideModel()

    val fillMaps = naObject.generateNAFillConditions()

    var buildStringMapCharacter = codeGen + "\n\t" + ".na.fill(Map("

    fillMaps.characterFillMap.foreach{ x =>
      buildStringMapCharacter += s""""${x._1}" -> "${x._2}","""
    }
    fillMaps.numericFillMap.foreach{ x=>
      buildStringMapCharacter += s""""${x._1}" -> ${x._2},"""
    }
    if(buildStringMapCharacter.takeRight(1) == ",") buildStringMapCharacter.dropRight(1)

    new NAFillConstructorReturn(modelType = modelType) {
      override def codeGen: String = buildStringMapCharacter + "))\n"
      override def data: DataFrame = fillMaps.filledData
    }

  }

  /**
    * Variance Filtering code gen
    * @param data Dataframe coming from the naFill package.
    * @param codeGen The status of the scala code gen up to this point
    * @return [String, Array[String] Returns the code gen after variance filtering, as well as the columns that have
    *         been dropped from variance filtering.
    */
  private def varianceFilteringConstructor(data: DataFrame, codeGen: String): VarianceFilterConstructorReturn = {

    val varianceFiltering = new VarianceFiltering(data)
      .setLabelCol(_mainConfig.labelCol)
      .setFeatureCol(_mainConfig.featuresCol)
      .setDateTimeConversionType(_mainConfig.dateTimeConversionType)

    val (varianceFilteredData, fieldsToDrop) = varianceFiltering.filterZeroVariance(_mainConfig.fieldsToIgnoreInVector)

    var codeGenVar = codeGen
    fieldsToDrop.foreach{x =>
      codeGenVar = codeGenVar + s"""\t.drop("$x")\n"""
    }

    new VarianceFilterConstructorReturn(fieldsToDrop = fieldsToDrop) {
      override def codeGen: String = codeGenVar
      override def data: DataFrame = varianceFilteredData
    }

  }

private def outlierFilterConstructor(data: DataFrame, codeGen: String): OutlierFilterConstructor = {
  
}

  /** OutlierFiltering */
  // return a list of column name, filter conditions.  Code gen a filter statement.

  /** CovarianceFiltering */
  // return a list of columns that should be removed. Code gen a filter statement.

  /** PearsonFiltering */
  // return a list of columns that should be removed. Code gen a filter statement.

  /** Feature Vector Creation */
  // Get the features that should be included in the vector after being removed from previous array. START PIPELINE

  /** Scaling */
  // retain the type of scaler used and params, add to pipeline object

  /** Model */
  // Get the Best HyperParams from tuning and generate a model pipeline for reproducability

  /** Model Serving */
  // Get the built model, put it in the pipeline object, and save the pipeline for batch inference.



  val codeGen = "val rawData = <load data here as DataFrame>"
  val naFillReturn: NAFillConstructorReturn = naFillConstructor(df, codeGen)
  val varianceFilterReturn: VarianceFilterConstructorReturn = varianceFilteringConstructor(
    naFillReturn.data, naFillReturn.codeGen)


  val outputCode: String =
    """
      |// Ensure that the label column is of DoubleType to use this code gen.
      |
      |val
      |
      |
      |
    """.stripMargin

}
