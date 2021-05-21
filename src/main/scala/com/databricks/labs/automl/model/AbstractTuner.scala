package com.databricks.labs.automl.model

import com.databricks.labs.automl.params.{TunerConfigBase, TunerOutputWithResults}
import org.apache.spark.sql.DataFrame

trait  AbstractTuner[A <: TunerConfigBase, B <: TunerOutputWithResults[A, C], C] {

  def postRunModeledHyperParams(paramsToTest: Array[A]): (Array[B], DataFrame)

  def evolveWithScoringDF(): (Array[B], DataFrame)

}
