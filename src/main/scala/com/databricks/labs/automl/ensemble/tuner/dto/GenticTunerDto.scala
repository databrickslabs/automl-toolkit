package com.databricks.labs.automl.ensemble.tuner.dto

import com.databricks.labs.automl.params.RandomForestModelsWithResults

case class GenticTunerDto(resultBuffer: Array[RandomForestModelsWithResults])
