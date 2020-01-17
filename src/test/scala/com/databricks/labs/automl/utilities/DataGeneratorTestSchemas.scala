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

case class VarianceTestSchema(a: Double,
                              b: Double,
                              c: Double,
                              d: Int,
                              label: Int,
                              automl_internal_id: Long)

case class PearsonTestSchema(positiveCorr1: Int,
                             positiveCorr2: Int,
                             noFilter1: Double,
                             noFilter2: Int,
                             label: Int,
                             automl_internal_id: Long)

case class PearsonRegressionTestSchema(positiveCorr1: Double,
                                       positiveCorr2: Double,
                                       positiveCorr3: Int,
                                       noFilter1: Double,
                                       noFilter2: Int,
                                       label: Double,
                                       automl_internal_id: Long)

case class FeatureCorrelationTestSchema(a1: Double,
                                        a2: Double,
                                        b1: Int,
                                        b2: Int,
                                        c1: Double,
                                        c2: Double,
                                        c3: Double,
                                        d1: Long,
                                        d2: Long,
                                        label: Double,
                                        automl_internal_id: Long)

case class CardinalityFilteringTestSchema(a: Double, b: Int, c: Long, d: String)

case class SanitizerSchema(a: Double,
                           b: Int,
                           c: Long,
                           d: String,
                           e: Int,
                           f: Boolean,
                           label: String,
                           automl_internal_id: Long)

case class SanitizerSchemaRegressor(a: Double,
                                    b: Int,
                                    c: Long,
                                    d: String,
                                    e: Int,
                                    f: Boolean,
                                    label: Double,
                                    automl_internal_id: Long)

case class KSampleSchema(a: Double,
                         b: Int,
                         c: Double,
                         label: Int,
                         automl_internal_id: Long)

case class FeatureInteractionSchema(a: Double,
                                    b: Double,
                                    c: Int,
                                    d: String,
                                    e: Int,
                                    f: String,
                                    label: Int,
                                    automl_internal_id: Long)
