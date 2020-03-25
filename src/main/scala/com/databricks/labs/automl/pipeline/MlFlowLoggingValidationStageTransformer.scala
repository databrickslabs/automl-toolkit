package com.databricks.labs.automl.pipeline

import com.databricks.labs.automl.exceptions.MlFlowValidationException
import com.databricks.labs.automl.utils.{AutoMlPipelineMlFlowUtils, WorkspaceDirectoryValidation}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * @author Jas Bali
  * A [[WithNoopsStage]] transformer stage that does MlFlow validation before
  * continuing with the rest of the stages. This should be added in the earliest stages of a
  * pipeline
  * @param uid
  */
class MlFlowLoggingValidationStageTransformer(override val uid: String)
  extends AbstractTransformer
  with DefaultParamsWritable
  with WithNoopsStage {

  def this() = {
    this(Identifiable.randomUID("MlFlowLoggingValidationStageTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setDebugEnabled(false)
  }

  final val mlFlowLoggingFlag: BooleanParam = new BooleanParam(this, "mlFlowLoggingFlag", "whether to log to MlFlow or not")

  final val mlFlowTrackingURI: Param[String] = new Param[String](this, "mlFlowTrackingURI", "MlFlow Tracking URI")

  final val mlFlowAPIToken: Param[String] = new Param[String](this, "mlFlowAPIToken", "MlFlow API token")

  final val mlFlowExperimentName: Param[String] = new Param[String](this, "mlFlowExperimentName", "MlFlow Experiment name")

  def setMlFlowLoggingFlag(value: Boolean): this.type = set(mlFlowLoggingFlag, value)

  def getMlFlowLoggingFlag: Boolean = $(mlFlowLoggingFlag)

  def setMlFlowTrackingURI(value: String): this.type = set(mlFlowTrackingURI, value)

  def getMlFlowTrackingURI: String = $(mlFlowTrackingURI)

  def setMlFlowAPIToken(value: String): this.type = set(mlFlowAPIToken, value)

  def getMlFlowAPIToken: String = $(mlFlowAPIToken)

  def setMlFlowExperimentName(value: String): this.type = set(mlFlowExperimentName, value)

  def getMlFlowExperimentName: String = $(mlFlowExperimentName)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    if (getMlFlowLoggingFlag) {
      try {
        val dirValidate = WorkspaceDirectoryValidation(
          getMlFlowTrackingURI,
          getMlFlowAPIToken,
          getMlFlowExperimentName
        )
        if (dirValidate) {
          val rgx = "(\\/\\w+$)".r
          val dir =
            rgx.replaceFirstIn(getMlFlowExperimentName, "")
          println(
            s"MLFlow Logging Directory confirmed accessible at: " +
              s"$dir"
          )
        }
      } catch {
        case exception: Exception => throw MlFlowValidationException("Failed to validate MLflow configuration", exception)
      }
    }
    dataset.toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): MlFlowLoggingValidationStageTransformer = defaultCopy(extra)
}

object MlFlowLoggingValidationStageTransformer extends DefaultParamsReadable[MlFlowLoggingValidationStageTransformer] {
  override def load(path: String): MlFlowLoggingValidationStageTransformer = super.load(path)
}
