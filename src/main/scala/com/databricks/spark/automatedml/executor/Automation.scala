package com.databricks.spark.automatedml.executor

import com.databricks.spark.automatedml.params.DataPrepConfig
import com.databricks.spark.automatedml.utils.AutomationTools
import org.apache.spark.sql.DataFrame

class Automation extends AutomationConfig with AutomationTools {

  require(_supportedModels.contains(_mainConfig.modelFamily))

  def dataPrep(data: DataFrame): (DataFrame, Array[String], String) = {

    val flagConfig = DataPrepConfig(
      naFillFlag = _mainConfig.naFillFlag,
      varianceFilterFlag = _mainConfig.varianceFilterFlag,
      outlierFilterFlag = _mainConfig.outlierFilterFlag,
      covarianceFilterFlag = _mainConfig.covarianceFilteringFlag,
      pearsonFilterFlag = _mainConfig.pearsonFilteringFlag,
      scalingFlag = _mainConfig.scalingFlag
    )

    new DataPrep(data).setDataPrepFlags(flagConfig).prepData()

  }

}
