package com.databricks.labs.automl.pipeline
import com.databricks.labs.automl.utils.AutoMlPipelineMlFlowUtils
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * A [[WithNoopsStage]] transformer stage that is helpful to repartition a
  * DataFrame coming out of any pipeline stage
  * @author Jas Bali
  * @param uid
  */
class RepartitionTransformer(override val uid: String)
      extends AbstractTransformer
      with WithNoopsStage
      with DefaultParamsWritable
      with HasDebug {

  def this() = {
    this(Identifiable.randomUID("RepartitionTransformer"))
    setAutomlInternalId(AutoMlPipelineMlFlowUtils.AUTOML_INTERNAL_ID_COL)
    setPartitionScaleFactor(1)
    setDebugEnabled(false)
  }

  final val partitionScaleFactor: IntParam = new IntParam(this, "partitionScaleFactor", "Scale factor of repartition (multiple of executors)")

  def setPartitionScaleFactor(value: Int): this.type = set(partitionScaleFactor, value)

  def getPartitionScaleFactor: Int = $(partitionScaleFactor)

  override def transformInternal(dataset: Dataset[_]): DataFrame = {
    val executors: Int = try {
      dataset.sparkSession.conf.get("spark.databricks.clusterUsageTags.clusterMaxWorkers").toInt
    } catch {
      case ex: Exception =>  dataset.rdd.getNumPartitions
    }
    dataset.repartition(executors * getPartitionScaleFactor).toDF()
  }

  override def transformSchemaInternal(schema: StructType): StructType = {
    assert(getPartitionScaleFactor > 0, "Repartition scale factor needs to be greater than 0. Default is 1.")
    schema
  }

  override def copy(extra: ParamMap): RepartitionTransformer = defaultCopy(extra)
}

object RepartitionTransformer extends DefaultParamsReadable[RepartitionTransformer] {
  override def load(path: String): RepartitionTransformer = super.load(path)
}
