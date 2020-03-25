package com.databricks.labs.automl.pipeline

/**
  * @author Jas Bali
  * Marker interface to signify a pipeline stage is only applicable for the training phase
  * This can be used to extract predict-only stages in a pipeline for debugging or logging purposes.
  * It is not intended to use this trait for any other purposes. To add default behavior to transformer,
  * look at [[AbstractTransformer]]
  *
  * An example [[IsTrainingStage]] is [[SyntheticFeatureGenTransformer]] which adds synthetic rows for
  * post model optimization process and is not required for inference
  */
trait IsTrainingStage {
}
