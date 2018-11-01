package com.databricks.spark.automatedml


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

class FeatureImportance(df: DataFrame, modelSelection: String, modelPayload: ModelsWithResults, importances: DataFrame)
  extends DataValidation with SparkSessionWrapper {


  private var _labelCol = "label"
  private var _featuresCol = "features"




  final val dfSchemaNames = df.schema.fieldNames


  def setLabelCol(value: String): this.type = {
    assert(dfSchemaNames.contains(_labelCol, s"DataFrame supplied does not contain label column '$value'"))
    _labelCol = value
    this
  }
  def setFeaturesCol(value: String): this.type = {
    assert(dfSchemaNames.contains(_featuresCol, s"DataFrame supplied does not contain label column '$value'"))
    _featuresCol = value
    this
  }


  def getLabelCol: String = _labelCol
  def getFeaturesCol: String = _featuresCol


  //TODO: Implement Bertrand's additive feature selection algorithm using information gain
  //TODO: Add in Linear Regression + Logistic Regression Support
  //TODO: Outlier Filtering based on algorithmic approach OR supplied fields
  //TODO: Implement pearson auto-correlation feature removal (automated and manual override mode)
  //TODO: Model report information (MLFlow + static table explaining what happened within the framework)

  def generateFrameReport(columns: Array[String], importances: Array[Double]): DataFrame = {
    import spark.sqlContext.implicits._
    sc.parallelize(columns zip importances).toDF("Columns", "Importances").orderBy($"Importances".desc)
  }

  def cleanupFieldArray(indexedFields: Array[(String, Int)]): List[(String, Int)] = {
    val cleanedBuffer = new ListBuffer[(String, Int)]
    indexedFields.map(x => {
      cleanedBuffer += ((x._1.split("_si$")(0), x._2))
    })
    cleanedBuffer.result
  }

  def reportFields(fieldIndexArray: Array[(String, Int)]) = {

    cleanupFieldArray(fieldIndexArray).foreach(x => {
      println(s"Column ${x._1} is feature ${x._2}")
    })
  }

  def reportImportances(assembledColumns: Array[String]):DataFrame = {

    val importances = modelSelection match {
      case "classifier" => modelPayload.model.asInstanceOf[RandomForestClassificationModel].featureImportances.toArray
      case "regressor" => modelPayload.model.asInstanceOf[RandomForestRegressionModel].featureImportances.toArray
      case _ => throw new UnsupportedOperationException(s"The model provided, '$modelSelection', is not supported.")
    }

    generateFrameReport(assembledColumns, importances)
      .withColumn("Importances", col("Importances") * 100)
      .withColumn("Columns", split(col("Columns"), "_si$")(0))
  }

  // TODO: FIX THIS MOVE IT AND MAKE IT SUPPORT CLASSIFIER AND REGRESSOR!!!!
  case class DecisionTreeConfig(impurity: String,
                                maxBins: Int,
                                maxDepth: Int,
                                minInfoGain: Double,
                                minInstancesPerNode: Int,
                                importanceCutoff: Double
                               )

  private def generateDecisionTree[A,B](decisionTreeConfig: DecisionTreeConfig) = {



    // pull this out and put a switch statement in for classifier vs regressor
    val treeClassifierModel = new DecisionTreeClassifier()
      .setLabelCol(_labelCol)
      .setFeaturesCol(_featuresCol)
      .setImpurity(decisionTreeConfig.impurity)
      .setMaxBins(decisionTreeConfig.maxBins)
      .setMaxDepth(decisionTreeConfig.maxDepth)
      .setMinInfoGain(decisionTreeConfig.minInfoGain)
      .setMinInstancesPerNode(decisionTreeConfig.minInstancesPerNode)

    val filteredData = filterData(df, importances, decisionTreeConfig.importanceCutoff)

    val (numericFields, categoricalFields) = extractTypes(filteredData, _labelCol)

    val (indexers, assembledColumns, assembler) = generateAssembly(numericFields, categoricalFields, _featuresCol)

    val pipeline = new Pipeline()
      .setStages(indexers :+ assembler :+ treeClassifierModel)

    val treeModel = pipeline.fit(filteredData)

    val decisionTreeModel = treeModel.stages.last.asInstanceOf[DecisionTreeClassificationModel]

    val indexedArray = assembledColumns.zipWithIndex

    (indexedArray, decisionTreeModel)

  }

  def filterData(rawData: DataFrame, reportResult: DataFrame, cutoff: Double) = {

    val filteredData = reportResult.filter(col("Importances") >= cutoff)
    val filteredFields = filteredData.select("Columns").as[String].collect

    assert(filteredFields.length > 0, "All feature fields have been filtered!  Retry with a lower threshold.")

    val fieldBuffer = new ArrayBuffer[String]

    filteredFields.map(x => fieldBuffer += x)
    fieldBuffer += "label"

    val fields = fieldBuffer.result.toArray

    rawData.select(fields.map(col):_*)

  }

  def generateDecisionText(decisionTreeModel: DecisionTreeClassificationModel, modelColumnArray: Array[(String, Int)]) = {
    val reparsedArray = new ArrayBuffer[(String, String)]
    cleanupFieldArray(modelColumnArray).toArray.map(x => {
      reparsedArray += (("feature " + x._2.toString, x._1))
    })
    val mappedArray = reparsedArray.result.toMap
    mappedArray.foldLeft(decisionTreeModel.toDebugString){case(body,(k,v)) => body.replaceAll(k,v)}
  }

}
