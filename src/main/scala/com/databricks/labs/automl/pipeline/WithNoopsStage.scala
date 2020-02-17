package com.databricks.labs.automl.pipeline


/**
  * Marker interface to signify any transformer extending this trait will not alter
  * an input dataset. This is only for the edge cases where it is required to do an external
  * Ops before pipeline execution can continue. An example would be to do Mlflow params Validation
  * before training continues. Helpful in scenarios where fail-fast feature is needed
  * Example transformers are [[DropTempTableTransformer]], [[MlFlowLoggingValidationStageTransformer]].
  *
  * NOTE: Noops implies no changes to the input Dataset, but the implementation can result in a change
  * to an external state
  * @author Jas Bali
  */
trait WithNoopsStage {
}
