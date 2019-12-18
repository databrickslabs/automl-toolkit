package com.databricks.labs.automl.sanitize.components

import com.databricks.labs.automl.{AbstractUnitSpec, DiscreteTestDataGenerator}

class ModelDetectionTest extends AbstractUnitSpec {

  it should "correctly identify a classification problem" in {

    val dat = DiscreteTestDataGenerator.generateNAFillData(100, 5)

    dat.show(50)

    val dat2 = DiscreteTestDataGenerator.reassignToNulls(dat)

    dat2.show(50)

  }

}
