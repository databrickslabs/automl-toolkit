package com.databricks.labs.automl.utils

import com.databricks.labs.automl.params.{Defaults, MainConfig}

object DefaultConfigAccessor extends Defaults {

  def getMainConfig: MainConfig = _mainConfigDefaults

  //TODO: finish this.

}
