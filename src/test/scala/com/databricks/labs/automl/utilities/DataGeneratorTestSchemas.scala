package com.databricks.labs.automl.utilities

case class SchemaNamesTypes(name: String, dataType: String)

case class ModelDetectionSchema(a: Double,
                                label: Double,
                                automl_internal_id: Long)

case class OutlierTestSchema(a: Double,
                             b: Double,
                             c: Double,
                             label: Int,
                             automl_internal_id: Long)

case class NaFillTestSchema(dblData: Double,
                            fltData: Float,
                            intData: Int,
                            ordinalIntData: Int,
                            strData: String,
                            boolData: Boolean,
                            dateData: String,
                            label: Int,
                            automl_internal_id: Long)
