package com.databricks.labs.automl.executor.config

import com.databricks.labs.automl.exploration.structures.FeatureImportanceConfig
import com.databricks.labs.automl.params._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.{read, writePretty}
import org.json4s.{Formats, FullTypeHints, NoTypeHints}

import scala.collection.mutable.ListBuffer

/**
  * @constructor Generate a configuration InstanceConfig for a given prediction type (either regressor or classifier)
  * @author Ben Wilson, Databricks
  * @param predictionType either 'regressor' or 'classifier', depending on the type of supervised ML needed for the task
  */
class GenericConfigGenerator(predictionType: String)
    extends ConfigurationDefaults {

  import PredictionType._

  private val familyType: PredictionType = predictionTypeEvaluator(
    predictionType
  )

  private var _genericConfig = genericConfig(familyType)

  /**
    * Setter
    *
    * @param value name of the Label column for the supervised learning task
    */
  def setLabelCol(value: String): this.type = {
    _genericConfig.labelCol = value
    this
  }

  /**
    * Setter
    *
    * @param value name of the feature vector to be used throughout the modeling process.
    */
  def setFeaturesCol(value: String): this.type = {
    _genericConfig.featuresCol = value
    this
  }

  /**
    * Setter
    *
    * @param value type of data to convert a datetime field to allowable values:
    *              "unix" - converts to a LongType for the number of milliseconds since Jan 1, 1970
    *              "split" - converts the aspects of the date into representative columns ->
    *               Year, Month, Day, Hour, Minute, Second
    * @throws IllegalArgumentException() if an invalid entry is made.
    */
  @throws(classOf[IllegalArgumentException])
  def setDateTimeConversionType(value: String): this.type = {
    validateMembership(
      value,
      allowableDateTimeConversionTypes,
      "DateTimeConversionType"
    )
    _genericConfig.dateTimeConversionType = value
    this
  }

  /**
    * Setter
    *
    * @param value Collection (Array) of fields that will be ignored throughout modeling and will not be included
    *              in feature vector operations.
    */
  def setFieldsToIgnoreInVector(value: Array[String]): this.type = {
    _genericConfig.fieldsToIgnoreInVector = value
    this
  }

  /**
    * Setter
    *
    * @param value Metric to be used to determine the 'best of' within generations of mutation.
    *              Allowable values for regressor: List("rmse", "mse", "r2", "mae")
    *              Allowable values for classifier: List("f1", "weightedPrecision", "weightedRecall",
    *              "accuracy", "areaUnderPR", "areaUnderROC")
    * @throws IllegalArgumentException() if an invalid entry is made.
    */
  @throws(classOf[IllegalArgumentException])
  def setScoringMetric(value: String): this.type = {
    familyType match {
      case Regressor =>
        validateMembership(
          value,
          allowableRegressionScoringMetrics,
          s"$predictionType Scoring Metric"
        )
      case Classifier =>
        validateMembership(
          value,
          allowableClassificationScoringMetrics,
          s"$predictionType Scoring Metric"
        )
    }
    _genericConfig.scoringMetric = value
    this
  }

  /**
    * Setter
    *
    * @param value Direction of optimization. Options:<br>
    *              <i>'maximize'</i> - will sort returned scores in descending order and take the top(n)<br>
    *              <i>'minimize'</i> - will sort returned scores in ascending order and take the top(n)
    * @throws IllegalArgumentException if an invalid entry is made.
    */
  @throws(classOf[IllegalArgumentException])
  def setScoringOptimizationStrategy(value: String): this.type = {
    validateMembership(
      value,
      allowableScoringOptimizationStrategies,
      "ScoringOptimizationStrategy"
    )
    _genericConfig.scoringOptimizationStrategy = value
    this
  }

  /**
    * Setter<br>
    *
    * Aids in creating multiple instances of a Generic Config (useful for Feature Importance usages)
    *
    * @param value an Instance of a GenericConfig Object
    */
  def setConfig(value: GenericConfig): this.type = {
    _genericConfig = value
    this
  }

  /**
    * Getter
    *
    * @return Currently assigned name of the label column for modeling.
    */
  def getLabelCol: String = _genericConfig.labelCol

  /**
    * Getter
    *
    * @return Currently assigned name of the feature column for the modeling vector.
    */
  def getFeaturesCol: String = _genericConfig.featuresCol

  /**
    * Getter
    *
    * @return Currently assigned setting for the datetime column conversion methodology.
    */
  def getDateTimeConversionType: String = _genericConfig.dateTimeConversionType

  /**
    * Getter
    *
    * @return A collection (default Empty Array) of fields that are to be ignored for the purposes of modeling.
    */
  def getFieldsToIgnoreInVector: Array[String] =
    _genericConfig.fieldsToIgnoreInVector

  /**
    * Getter
    *
    * @return Currently assigned setting for the metric to be used for determining quality of models for subsequent
    *         optimization generations / iterations.
    */
  def getScoringMetric: String = _genericConfig.scoringMetric

  /**
    * Getter
    *
    * @return Currently assigned setting for the direction of sorting for the provided scoringMetric value
    *         (either 'minimize' or 'maximize')
    */
  def getScoringOptimizationStrategy: String =
    _genericConfig.scoringOptimizationStrategy

  /**
    * Main Method accessor to return the GenericConfig current state.
    *
    * @return :GenericConfig type objects of the results of setter usage.
    */
  def getConfig: GenericConfig = _genericConfig

}

object GenericConfigGenerator {

  /**
    * Companion object apply generator
    *
    * @param predictionType the type of modeling desired: 'regressor' or 'classifier'
    * @return Instance of the GenericConfigGenerator with defaults applied.
    */
  def apply(predictionType: String): GenericConfigGenerator =
    new GenericConfigGenerator(predictionType)

  /**
    * Helper method that allows for default settings for a classifier to be used and generated
    *
    * @example ```
    *         val defaultClassifierGenericConfig = GenericConfigGenerator.generateDefaultClassifierConfig
    *         ```
    * @return GenericConfig Object, setup for classifiers.
    *
    */
  def generateDefaultClassifierConfig: GenericConfig =
    new GenericConfigGenerator("classifier").getConfig

  /**
    * Helper method that allows for default settings for a regressor to be used and generated
    *
    * @example ```
    *          val defaultRegressirGenericConfig = GenericConfigGenerator.generateDefaultRegressorConfig
    *          ```
    * @return GenericConfig Object, setup for regressors.
    */
  def generateDefaultRegressorConfig: GenericConfig =
    new GenericConfigGenerator("regressor").getConfig

}

/**
  * Main Configuration Generator utility class, used for generating a modeling configuration to execute the autoML
  * framework.
  *
  * @since 0.5
  * @author Ben Wilson, Databricks
  * @param modelFamily The model family that is desired to be run (e.g. 'RandomForest')
  *                    Allowable Options:
  *                     "Trees", "GBT", "RandomForest", "LinearRegression", "LogisticRegression", "XGBoost", "MLPC",
  *                     "SVM"
  * @param predictionType The modeling type that is desired to be run (e.g. 'classifier')
  *                       Allowable Options:
  *                       "classifier" or "regressor"
  * @param genericConfig Configuration object from GenericConfigGenerator
  */
class ConfigurationGenerator(modelFamily: String,
                             predictionType: String,
                             var genericConfig: GenericConfig)
    extends ConfigurationDefaults {

  import FamilyValidator._
  import ModelSelector._

  private val modelType: ModelSelector =
    modelTypeEvaluator(modelFamily, predictionType)
  private val family: FamilyValidator = familyTypeEvaluator(modelFamily)

  // Default configuration generation

  private var _instanceConfig = instanceConfig(modelFamily, predictionType)

  _instanceConfig.genericConfig = genericConfig

  /**
    * Helper method for copying a pre-defined InstanceConfig to a new instance.
    *
    * @param value InstanceConfig object
    */
  def setConfig(value: InstanceConfig): this.type = {
    _instanceConfig = value
    this
  }

  //Switch Config
  /**
    * Boolean switch for turning on naFill actions
    *
    * @note Default: On
    * @note HIGHLY RECOMMENDED TO LEAVE ON.
    */
  def naFillOn(): this.type = {
    _instanceConfig.switchConfig.naFillFlag = true
    this
  }

  /**
    * Boolean switch for turning off naFill actions
    *
    * @note Default: On
    * @note HIGHLY RECOMMENDED TO NOT TURN OFF
    */
  def naFillOff(): this.type = {
    _instanceConfig.switchConfig.naFillFlag = false
    this
  }

  /**
    * Boolean switch for setting the state of naFillFlag
    *
    * @param value Boolean
    *              (whether to execute filling of na values on the DataFrame's non-ignored fields)
    */
  def setNaFillFlag(value: Boolean): this.type = {
    _instanceConfig.switchConfig.naFillFlag = value
    this
  }

  /**
    * Boolean switch for turning variance filtering on
    *
    * @note Default: On
    */
  def varianceFilterOn(): this.type = {
    _instanceConfig.switchConfig.varianceFilterFlag = true
    this
  }

  /**
    * Boolean switch for turning variance filtering off
    *
    * @note Default: On
    */
  def varianceFilterOff(): this.type = {
    _instanceConfig.switchConfig.varianceFilterFlag = false
    this
  }

  /**
    * Boolean switch for setting the state of varianceFilterFlag
    *
    * @param value Boolean
    *              (whether or not to filter out fields from the feature vector that all have the same value)
    */
  def setVarianceFilterFlag(value: Boolean): this.type = {
    _instanceConfig.switchConfig.varianceFilterFlag = value
    this
  }

  /**
    * Boolean switch for turning outlier filtering on
    *
    * @note Default: Off
    */
  def outlierFilterOn(): this.type = {
    _instanceConfig.switchConfig.outlierFilterFlag = true
    this
  }

  /**
    * Boolean switch for turning outlier filtering off
    *
    * @note Default: Off
    */
  def outlierFilterOff(): this.type = {
    _instanceConfig.switchConfig.outlierFilterFlag = false
    this
  }

  /**
    * Boolean switch for setting the state of outlierFilterFlag
    *
    * @param value Boolean
    */
  def setOutlierFilterFlag(value: Boolean): this.type = {
    _instanceConfig.switchConfig.outlierFilterFlag = value
    this
  }

  /**
    * Boolean switch for turning Pearson filtering on
    *
    * @note Default: Off
    */
  def pearsonFilterOn(): this.type = {
    _instanceConfig.switchConfig.pearsonFilterFlag = true
    this
  }

  /**
    * Boolean switch for turning Pearson filtering off
    *
    * @note Default: Off
    */
  def pearsonFilterOff(): this.type = {
    _instanceConfig.switchConfig.pearsonFilterFlag = false
    this
  }

  /**
    * Boolean switch for setting the state of pearsonFilterFlag
    *
    * @param value Boolean
    */
  def setPearsonFilterFlag(value: Boolean): this.type = {
    _instanceConfig.switchConfig.pearsonFilterFlag = value
    this
  }

  /**
    * Boolean switch for turning Covariance filtering on
    *
    * @note Default: Off
    */
  def covarianceFilterOn(): this.type = {
    _instanceConfig.switchConfig.covarianceFilterFlag = true
    this
  }

  /**
    * Boolean switch for turning Covariance filtering off
    *
    * @note Default: Off
    */
  def covarianceFilterOff(): this.type = {
    _instanceConfig.switchConfig.covarianceFilterFlag = false
    this
  }

  /**
    * Boolean switch for setting the state of covarianceFilterFlag
    *
    * @param value Boolean
    */
  def setCovarianceFilterFlag(value: Boolean): this.type = {
    _instanceConfig.switchConfig.covarianceFilterFlag = value
    this
  }

  /**
    * Boolean switch for turning One Hot Encoding of string and character features on
    *
    * @note Default: Off for Tree based algorithms, On for all others.
    * @note Turning One Hot Encoding on for a tree-based algorithm (XGBoost, RandomForest, Trees, GBT) is not
    *       recommended.  Introducing synthetic dummy variables in a tree algorithm will force the creation of
    *       sparse tree splits.
    * @see See [[https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769]]
    *      for a full explanation.
    */
  def oneHotEncodeOn(): this.type = {
    family match {
      case Trees =>
        println(
          "WARNING! OneHotEncoding set on a trees algorithm will likely create a poor model.  " +
            "Proceed at your own risk!"
        )
      case _ => None
    }
    _instanceConfig.switchConfig.oneHotEncodeFlag = true
    this
  }

  /**
    * Boolean switch for turning off One Hot Encoding
    *
    * @note Default: Off for Tree based algorithms, On for all others.
    */
  def oneHotEncodeOff(): this.type = {
    _instanceConfig.switchConfig.oneHotEncodeFlag = false
    this
  }

  /**
    * Boolean switch for setting the state of oneHotEncodeFlag
    *
    * @param value Boolean
    */
  def setOneHotEncodeFlag(value: Boolean): this.type = {
    if (value) oneHotEncodeOn()
    else oneHotEncodeOff()
    this
  }

  /**
    * Boolean switch for turning scaling On
    *
    * @note Default: Off for Tree based algorithms, On for all others.
    * @note For Tree based algorithms (RandomForest, XGBoost, GBT, Trees), it is not necessary (and can adversely
    *       affect the model performance) that this be turned on.
    */
  def scalingOn(): this.type = {
    _instanceConfig.switchConfig.scalingFlag = true
    this
  }

  /**
    * Boolean switch for turning scaling Off
    *
    * @note Default: Off for Tree based algorithms, On for all others.
    */
  def scalingOff(): this.type = {
    _instanceConfig.switchConfig.scalingFlag = false
    this
  }

  /**
    * Boolean switch for setting the state of the scalingFlag
    *
    * @param value Boolean
    */
  def setScalingFlag(value: Boolean): this.type = {
    _instanceConfig.switchConfig.scalingFlag = value
    this
  }

  /**
    * Boolean switch for setting featureInteraction on.
    * This setting will, in conjunction with the settings for featureInteraction elements in the config,
    * perform pair-wise product interactions of all elements of the feature vector, retaining either all or some
    * of those interactions for inclusion to the feature vector.
    * For classification tasks, InformationGain is used as the metric to compare inclusion (for modes other than 'all')
    * For regression tasks, Variance is used as the metric.
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def featureInteractionOn(): this.type = {
    _instanceConfig.switchConfig.featureInteractionFlag = true
    this
  }

  /**
    * Boolean switch for turning featureInteraction off
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def featureInteractionOff(): this.type = {
    _instanceConfig.switchConfig.featureInteractionFlag = false
    this
  }

  /**
    * Setter for defining the state of the featureInteractionFlag
    * @param value Boolean on/off
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def setFeatureInteractionFlag(value: Boolean): this.type = {
    _instanceConfig.switchConfig.featureInteractionFlag = value
    this
  }

  /**
    * Boolean switch for setting the Data Prep Caching On
    *
    * @note Default: On
    * @note Depending on the size and partitioning of the data set, caching may or may not improve performance.
    */
  def dataPrepCachingOn(): this.type = {
    _instanceConfig.switchConfig.dataPrepCachingFlag = true
    this
  }

  /**
    * Boolean switch for setting the Data Prep Caching Off
    *
    * @note Default: On
    * @note Depending on the size and partitioning of the data set, caching may or may not improve performance.
    */
  def dataPrepCachingOff(): this.type = {
    _instanceConfig.switchConfig.dataPrepCachingFlag = false
    this
  }

  /**
    * Boolean switch for setting the state of DataPrepCachingFlag
    *
    * @param value Boolean
    */
  def setDataPrepCachingFlag(value: Boolean): this.type = {
    _instanceConfig.switchConfig.dataPrepCachingFlag = value
    this
  }

  /**
    * Boolean switch for setting Auto Stopping On
    *
    * @note Default: Off
    * @note Early stopping will invalidate the progress measurement system (due to non-determinism)
    *       Early termination will not occur immediately.  Futures objects already committed will continue to run, but
    *       no new actions will be enqueued when a stopping criteria is met.
    */
  def autoStoppingOn(): this.type = {
    _instanceConfig.switchConfig.autoStoppingFlag = true
    this
  }

  /**
    * Boolean switch for setting Auto Stopping Off
    *
    * @note Default: Off
    */
  def autoStoppingOff(): this.type = {
    _instanceConfig.switchConfig.autoStoppingFlag = false
    this
  }

  /**
    * Boolean switch for setting the state of autoStoppingFlag
    *
    * @param value Boolean
    */
  def setAutoStoppingFlag(value: Boolean): this.type = {
    _instanceConfig.switchConfig.autoStoppingFlag = value
    this
  }

  def setPipelineDebugFlag(value: Boolean): this.type = {
    _instanceConfig.switchConfig.pipelineDebugFlag = value
    this
  }

  def pipelineDebugFlagOn(value: Boolean): this.type = {
    _instanceConfig.switchConfig.pipelineDebugFlag = true
    this
  }

  def pipelineDebugFlagOff(value: Boolean): this.type = {
    _instanceConfig.switchConfig.pipelineDebugFlag = false
    this
  }

  // Feature Engineering Config

  /**
    * Setter for defining the number of concurrent threads allocated to performing asynchronous data prep tasks within
    * the feature engineering aspect of this application.
    * @param value Int: A value that must be greater than zero.
    * @note This value has an upper limit, depending on driver size, that will restrict the efficacy of the asynchronous
    *       tasks within the pool.  Setting this too high may cause cluster instability.
    * @author Ben Wilson, Databricks
    * @since 0.6.0
    * @throws IllegalArgumentException if a value less than or equal to zero is supplied.
    */
  @throws(classOf[IllegalArgumentException])
  def setDataPrepParallelism(value: Int): this.type = {

    require(value > 0, s"DataPrepParallelism must be greater than zero.")
    _instanceConfig.featureEngineeringConfig.dataPrepParallelism = value
    this
  }

  /**
    * Setter
    * Specifies the behavior of the naFill algorithm for numeric (continuous) fields.<br>
    * Values that are generated as potential fill candidates are set according to the available statistics that are
    * calculated from a df.summary() method.<br>
    * Available options are:<br>
    *     <i>"min", "25p", "mean", "median", "75p", or "max"</i>
    *
    * @param value String: member of allowable list.
    * @note Default: "mean"
    * @throws IllegalArgumentException if an invalid entry is made.
    */
  @throws(classOf[IllegalArgumentException])
  def setFillConfigNumericFillStat(value: String): this.type = {
    validateMembership(
      value,
      allowableNumericFillStats,
      "FillConfigNumericFillStat"
    )
    _instanceConfig.featureEngineeringConfig.numericFillStat = value
    this
  }

  /**
    * Setter
    * Specifies the behavior of the naFill algorithm for character (String, Char, Boolean, Byte, etc.) fields.
    * Generated through a df.summary() method<br>
    * Available options are:<br>
    *     <i>"min"</i> (least frequently occurring value)<br>
    *       or<br>
    *     <i>"max"</i> (most frequently occurring value)
    *
    * @param value String: member of allowable list
    * @note Default: "max"
    * @throws IllegalArgumentException if an invalid entry is made.
    */
  @throws(classOf[IllegalArgumentException])
  def setFillConfigCharacterFillStat(value: String): this.type = {
    validateMembership(
      value,
      allowableCharacterFillStats,
      "FillConfigCharacterFillStat"
    )
    _instanceConfig.featureEngineeringConfig.characterFillStat = value
    this
  }

  /**
    * Setter<br>
    * The threshold value that is used to detect, based on the supplied labelCol, the cardinality of the label through
    * a .distinct().count() being issued to the label column.  Values from this cardinality determination that are
    * above this setter's value will be considered to be a Regression Task, those below will be considered a
    * Classification Task.
    *
    * @note In the case of exceptions being thrown for incorrect type (detected a classifier, but intended usage is for
    *       a regression, lower this value.  Conversely, if a classification problem has a significant number of
    *       classes, above the default threshold of this setting (50), increase this value.)
    * @param value Int: Threshold value for the labelCol cardinality check.  Values above this setting will be
    *              determined to be a regression task; below to be a classification task.
    * @note Default: 50
    */
  @deprecated(
    "This setter, and the logic underlying it for automatically detecting modeling type, will be removed" +
      "in future versions, as it is now required to be specified for utilizing a ConfigurationGenerator Object."
  )
  def setFillConfigModelSelectionDistinctThreshold(value: Int): this.type = {
    _instanceConfig.featureEngineeringConfig.modelSelectionDistinctThreshold =
      value
    this
  }

  /**
    * Setter
    * <p>Configures the tails of a distribution to filter out, along with the ntile settings defined in:
    *   .setOutlierLowerFilterNTile() and/or .setOutlierUpperFilterNTile()
    * <p>Available Modes:<br>
    *   <i>"lower"</i> -> filters out rows from the data that are below the value set in
    *   ```.setOutlierLowerFilterNTile()```<br>
    *   <i>"upper"</i> -> filter out rows from the data that are above the the value set in
    *   ```.setOutlierUpperFilterNTile()```<br>
    *   <i>"both"</i> -> two-tailed filter that combines both an "upper" and "lower" filter.<br>
    *
    * </p>
    * </p>
    *
    * @param value String: Tailed direction setting for outlier filtering.
    * @note Default: "both"
    * @note This filter action is disabled by default.  Before enabling, please ensure the fields to be filtered are
    *       adequately reflected in the ```.setOutlierFieldsToIgnore()``` inverse selection, as well as verifying the
    *       general distribution of the fields that have outlier data in order to select an appropriate NTile value.
    *       <u>This feature should only be supplied in rare instances and a full understanding of the impacts that this
    *       filter may have should be understood before enabling it.</u>
    */
  def setOutlierFilterBounds(value: String): this.type = {
    validateMembership(
      value,
      allowableOutlierFilterBounds,
      "OutlierFilterBounds"
    )
    _instanceConfig.featureEngineeringConfig.outlierFilterBounds = value
    this
  }

  /**
    *Setter<br>
    * Defines the NTILE value of the distributions of feature fields below which rows that fall beneath this value will
    * be filtered from the data.
    *
    * @param value Double: Lower Threshold boundary NTILE for Outlier Filtering
    * @note Only used if Outlier filtering is set to 'On' and Filter Direction is either 'both' or 'lower'
    * @throws IllegalArgumentException if the value supplied is outside of the Range(0.0,1.0)
    */
  @throws(classOf[IllegalArgumentException])
  def setOutlierLowerFilterNTile(value: Double): this.type = {
    zeroToOneValidation(value, "OutlierLowerFilterNTile")
    _instanceConfig.featureEngineeringConfig.outlierLowerFilterNTile = value
    this
  }

  /**
    * Setter<br>
    *   Defines the NTILE value of the distributions of feature fields above which rows that fall above this value will
    *   be filtered from the data
    *
    * @param value Double: Upper Threshold boundary NTILE value for Outlier Filtering
    * @note Only used if Outlier filtering is set to 'On' and Filter Direction is either 'both' or 'upper'
    * @throws IllegalArgumentException if the value supplied is outside of the Range(0.0,1.0)
    */
  @throws(classOf[IllegalArgumentException])
  def setOutlierUpperFilterNTile(value: Double): this.type = {
    zeroToOneValidation(value, "OutlierUpperFilterNTile")
    _instanceConfig.featureEngineeringConfig.outlierUpperFilterNTile = value
    this
  }

  /**
    *Setter<br>
    *   Defines the precision (RSD) in which each field's cardinality is calculated through the use of
    *   ```approx_count_distinct``` SparkSQL function.  Lower values specify higher accuracy, but consume
    *   more computational resources.
    *
    * @param value Double: In range of 0.0, 1.0
    * @note A Value of 0.0 will be an exact computation of distinct values.  Therefore, all data must be shuffled,
    *       which is an expensive task.
    * @see [[https://en.wikipedia.org/wiki/Coefficient_of_variation]] for explanation of RSD
    * @throws IllegalArgumentException if the value supplied is outside of the Range(0.0, 1.0)
    */
  @throws(classOf[IllegalArgumentException])
  def setOutlierFilterPrecision(value: Double): this.type = {
    zeroToOneValidation(value, "OutlierFilterPrecision")
    if (value == 0.0)
      println(
        "Warning! Precision of 0 is an exact calculation of quantiles and may not be performant!"
      )
    _instanceConfig.featureEngineeringConfig.outlierFilterPrecision = value
    this
  }

  /**
    * Setter<br>
    *   Defines the determination of whether to classify a numeric field as ordinal (categorical) or
    *   continuous.
    *
    * @param value Int: Threshold for distinct counts within a numeric feature field.
    * @note Continuous data fields are eligible for outlier filtering.  Categorical fields are not, and if below
    *       cardinality thresholds set by this value setter, those fields will be ignored by the filtering action.
    */
  def setOutlierContinuousDataThreshold(value: Int): this.type = {
    if (value < 50)
      println(
        "Warning! Values less than 50 may indicate ordinal (categorical numeric) data!"
      )
    _instanceConfig.featureEngineeringConfig.outlierContinuousDataThreshold =
      value
    this
  }

  /**
    * Setter<br>
    *   Defines an Array of fields to be ignored from outlier filtering.
    *
    * @param value Array[String]: field names to be ignored from outlier filtering.
    */
  def setOutlierFieldsToIgnore(value: Array[String]): this.type = {
    _instanceConfig.featureEngineeringConfig.outlierFieldsToIgnore = value
    this
  }

  /**
    * Setter<br>
    *   Selection for filter statistic to be used in Pearson Filtering.<br>
    *     Available modes: "pvalue", "degreesFreedom", or "pearsonStat"
    *
    * @note Default: pearsonStat
    * @param value String: one of available modes.
    * @throws IllegalArgumentException if the value provided is not in available modes list.
    */
  @throws(classOf[IllegalArgumentException])
  def setPearsonFilterStatistic(value: String): this.type = {
    validateMembership(
      value,
      allowablePearsonFilterStats,
      "PearsonFilterStatistic"
    )
    _instanceConfig.featureEngineeringConfig.pearsonFilterStatistic = value
    this
  }

  /**
    * Setter<br>
    * Controls which direction of correlation values to filter out.  Allowable modes: <br>
    *   "greater" or "lesser"
    *
    * @note Default: greater
    * @param value String: one of available modes
    * @throws IllegalArgumentException if the value provided is not in available modes list.
    */
  @throws(classOf[IllegalArgumentException])
  def setPearsonFilterDirection(value: String): this.type = {
    validateMembership(
      value,
      allowablePearsonFilterDirections,
      "PearsonFilterDirection"
    )
    _instanceConfig.featureEngineeringConfig.pearsonFilterDirection = value
    this
  }

  /**
    * Setter <br>
    *   Controls the Pearson manual filter value, if the PearsonFilterMode is set to "manual"<br>
    *
    *     @example with .setPearsonFilterMode("manual") and .setPearsonFilterDirection("greater") <br>
    *              the removal of fields that have a pearson correlation coefficient result above this <br>
    *              value will be dropped from modeling runs.
    * @param value Double: A value that is used as a cut-off point to filter fields whose correlation statistic is
    *              either above or below will be culled from the feature vector.
    */
  def setPearsonFilterManualValue(value: Double): this.type = {
    _instanceConfig.featureEngineeringConfig.pearsonFilterManualValue = value
    this
  }

  /**
    * Setter <br>
    *   Controls whether to use "auto" mode (using the PearsonAutoFilterNTile) or "manual" mode (using the <br>
    *     PearsonFilterManualValue) to cull fields from the feature vector.
    *
    * @param value String: either "auto" or "manual"
    * @note Default: "auto"
    * @throws IllegalArgumentException if the value provided is not in available modes list (auto and manual)
    */
  @throws(classOf[IllegalArgumentException])
  def setPearsonFilterMode(value: String): this.type = {
    validateMembership(value, allowablePearsonFilterModes, "PearsonFilterMode")
    _instanceConfig.featureEngineeringConfig.pearsonFilterMode = value
    this
  }

  /**
    * Setter <br>
    *   Provides the ntile threshold above or below which (depending on PearsonFilterDirection setting) fields will<br>
    *     be removed, depending on the distribution of pearson statistics from all feature columns.
    *
    * @note WARNING - this feature is ONLY recommended to be used for exploratory development work.
    * @note Default: 0.75 (Q3)
    * @param value Double: In range of (0.0, 1.0)
    * @throws IllegalArgumentException if the value provided is outside of the range of (0.0, 1.0)
    */
  @throws(classOf[IllegalArgumentException])
  def setPearsonAutoFilterNTile(value: Double): this.type = {
    zeroToOneValidation(value, "PearsonAutoFilterNTile")
    _instanceConfig.featureEngineeringConfig.pearsonAutoFilterNTile = value
    this
  }

  /**
    * Setter<br>
    *   Covariance Cutoff for specifying the feature-to-feature correlation statistic lower cutoff boundary
    *
    * @example For feature columns A, B, and C, if A->B is 0.02, A->C is 0.1, B->C is 0.85, with a value set of 0.05,
    *          <br> Column A would be removed from the feature vector for having a low value of the correlation
    *          statistic.
    * @param value Double: Threshold Cutoff Value
    * @note Default: -0.99
    * @note WARNING This setting is not recommended to be used in a production use case and is only potentially
    *       useful for data exploration and experimentation.
    * @note WARNING the lower threshold boundary for correlation is less frequently used.  Filtering of auto-correlated
    *       features is done primarily through .setCovarianceCutoffHigh values lower than the default of 0.99
    * @throws IllegalArgumentException if the value is <= -1.0
    */
  @throws(classOf[IllegalArgumentException])
  def setCovarianceCutoffLow(value: Double): this.type = {
    require(
      value > -1.0,
      s"Covariance Cutoff Low value $value is outside of allowable range.  Value must be " +
        s"greater than -1.0."
    )
    _instanceConfig.featureEngineeringConfig.covarianceCorrelationCutoffLow =
      value
    this
  }

  /**
    * Setter<br>
    *   Covariance Cutoff for specifying the feature-to-feature correlation statistic upper cutoff boundary
    *
    * @example For feature columns A, B, and C, if A<->B is 0.02, A<->C is 0.1, B<->C is 0.85, with a value set of 0.8,
    *          <br> Column C would be removed from the feature vector for having a high value of the correlation
    *          statistic.
    * @param value Double: Threshold Cutoff Value
    * @note Default: 0.99
    * @note WARNING This setting is not recommended to be used in a production use case and is only potentially
    *       useful for data exploration and experimentation.
    * @throws IllegalArgumentException if the value is <= -1.0
    */
  @throws(classOf[IllegalArgumentException])
  def setCovarianceCutoffHigh(value: Double): this.type = {
    require(
      value < 1.0,
      s"Covariance Cutoff High value $value is outside of allowable range.  Value must be " +
        s"less than 1.0."
    )
    _instanceConfig.featureEngineeringConfig.covarianceCorrelationCutoffHigh =
      value
    this
  }

  def setScalingType(value: String): this.type = {
    validateMembership(value, allowableScalers, "ScalingType")
    _instanceConfig.featureEngineeringConfig.scalingType = value
    this
  }

  def setScalingMin(value: Double): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingMin = value
    this
  }

  def setScalingMax(value: Double): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingMax = value
    this
  }

  def setScalingStandardMeanFlagOn(): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingStandardMeanFlag = true
    this
  }

  def setScalingStandardMeanFlagOff(): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingStandardMeanFlag = false
    this
  }

  def setScalingStandardMeanFlag(value: Boolean): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingStandardMeanFlag = value
    this
  }

  def setScalingStdDevFlagOn(): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingStdDevFlag = true
    this
  }

  def setScalingStdDevFlagOff(): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingStdDevFlag = false
    this
  }

  def setScalingStdDevFlag(value: Boolean): this.type = {
    _instanceConfig.featureEngineeringConfig.scalingStdDevFlag = value
    this
  }

  def setScalingPNorm(value: Double): this.type = {
    require(
      value >= 1.0,
      s"pNorm value: $value is invalid. Value must be greater than or equal to 1.0."
    )
    _instanceConfig.featureEngineeringConfig.scalingPNorm = value
    this
  }

  /**
    * Setter for determining the mode of operation for inclusion of interacted features.
    * Modes are:
    *   - all -> Includes all interactions between all features (after string indexing of categorical values)
    *   - optimistic -> If the Information Gain / Variance, as compared to at least ONE of the parents of the interaction
    *       is above the threshold set by featureInteractionTargetInteractionPercentage
    *         (e.g. if IG of left parent is 0.5 and right parent is 0.9, with threshold set at 10, if the interaction
    *         between these two parents has an IG of 0.42, it would be rejected, but if it was 0.46, it would be kept)
    *   - strict -> the threshold percentage must be met for BOTH parents.
    *         (in the above example, the IG for the interaction would have to be > 0.81 in order to be included in
    *         the feature vector).
    * @param value String -> one of: 'all', 'optimistic', or 'strict'
    * @throws IllegalArgumentException if the specified value submitted is not permitted
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[IllegalArgumentException])
  def setFeatureInteractionRetentionMode(value: String): this.type = {
    validateMembership(
      value,
      allowableFeatureInteractionModes,
      "featureInteractionRetentionMode"
    )
    _instanceConfig.featureEngineeringConfig.featureInteractionRetentionMode =
      value
    this
  }

  /**
    * Setter for determining the behavior of continuous feature columns.  In order to calculate Entropy for a continuous
    * variable, the distribution must be converted to nominal values for estimation of per-split information gain.
    * This setting defines how many nominal categorical values to create out of a continuously distributed feature
    * in order to calculate Entropy.
    * @param value Int -> must be greater than 1
    * @throws IllegalArgumentException if the value specified is <= 1
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def setFeatureInteractionContinuousDiscretizerBucketCount(
    value: Int
  ): this.type = {
    require(
      value > 1,
      s"FeatureInteractionContinuousDiscretizerBucketCount must be greater than 1."
    )
    _instanceConfig.featureEngineeringConfig.featureInteractionContinuousDiscretizerBucketCount =
      value
    this
  }

  /**
    * Setter for configuring the concurrent count for scoring of feature interaction candidates.
    * Due to the nature of these operations, the configuration here may need to be set differently to that of
    * the modeling and general feature engineering phases of the toolkit.  This is highly dependent on the row
    * count of the data set being submitted.
    * @param value Int -> must be greater than 0
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    * @throws IllegalArgumentException if the value is < 1
    */
  @throws(classOf[IllegalArgumentException])
  def setFeatureInteractionParallelism(value: Int): this.type = {
    require(
      value >= 1,
      s"FeatureInteractionParallelism must be set to a value >= 1."
    )
    _instanceConfig.featureEngineeringConfig.featureInteractionParallelism =
      value
    this
  }

  /**
    * Setter for establishing the minimum acceptable InformationGain or Variance allowed for an interaction
    * candidate based on comparison to the scores of its parents.
    * @param value Double in range of -inf -> inf
    * @since 0.6.2
    * @author Ben Wilson, Databricks
    */
  def setFeatureInteractionTargetInteractionPercentage(
    value: Double
  ): this.type = {
    _instanceConfig.featureEngineeringConfig.featureInteractionTargetInteractionPercentage =
      value
    this
  }

  def setFeatureImportanceCutoffType(value: String): this.type = {
    validateMembership(
      value,
      allowableFeatureImportanceCutoffTypes,
      "FeatureImportanceCufoffType"
    )
    _instanceConfig.featureEngineeringConfig.featureImportanceCutoffType = value
    this
  }

  def setFeatureImportanceCutoffValue(value: Double): this.type = {
    _instanceConfig.featureEngineeringConfig.featureImportanceCutoffValue =
      value
    this
  }

  def setDataReductionFactor(value: Double): this.type = {
    zeroToOneValidation(value, "DateReductionFactor")
    _instanceConfig.featureEngineeringConfig.dataReductionFactor = value
    this
  }

  /**
    * Setter switch for turning cardinality switch on
    * This switch is intended to set whether the a cardinality check is performed on StringIndexed columns
    *
    * @note Default: true
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  def cardinalitySwitchOn(): this.type = {
    _instanceConfig.featureEngineeringConfig.cardinalitySwitch = true
    this
  }

  /**
    * Setter switch for turning cardinality switch off.
    *
    * @note Not recommended for exploratory data set features.
    * @note Default: true
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  def cardinalitySwitchOff(): this.type = {
    _instanceConfig.featureEngineeringConfig.cardinalitySwitch = false
    this
  }

  /**
    * Setter for direct override of the cardinality switch
    *
    * @note Default: true
    * @param value: Boolean
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  def setFillConfigCardinalitySwitch(value: Boolean): this.type = {
    _instanceConfig.featureEngineeringConfig.cardinalitySwitch = value
    this
  }

  /**
    * Setter for specifying the mode of cardinality checking [either "approx" for approximate distinct or "exact"]
    *
    * @param value String: either "approx" or "exact"
    * @note Default - exact
    * @throws IllegalArgumentException if a mode other than exact or approx is specified.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[IllegalArgumentException])
  def setFillConfigCardinalityType(value: String): this.type = {
    validateMembership(
      value,
      allowableCardinalilties,
      "fillConfigCardinalityType"
    )
    _instanceConfig.featureEngineeringConfig.cardinalityType = value
    this
  }

  /**
    * Setter for overriding the default cardinality limit when validating whether a field should be considered for
    * OneHotEncoding or StringIndexing
    *
    * @param value Int: The value at above which a field will be declared to be of too high a cardinality for StringIndexing or OneHotEncoding
    * @note Default: 20
    * @throws java.lang.IllegalArgumentException if the number is <= to 0
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[IllegalArgumentException])
  def setFillConfigCardinalityLimit(value: Int): this.type = {
    require(value > 0, s"Cardinality limit must be greater than 0")
    _instanceConfig.featureEngineeringConfig.cardinalityLimit = value
    this
  }

  /**
    * Setter for defining the precision calculation when in "approx" mode for cardinalityType.  Must be in range 0 -> 1
    *
    * @param value Double: The precision for approximate distinct calculations for cardinality purposes
    * @throws java.lang.IllegalArgumentException if the Double supplied is outside of the range of 0 -> 1
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[IllegalArgumentException])
  def setFillConfigCardinalityPrecision(value: Double): this.type = {
    require(value >= 0.0, s"Precision must be greater than or equal to 0.")
    require(value <= 1.0, s"Precision must be less than or equal to 1.")
    _instanceConfig.featureEngineeringConfig.cardinalityPrecision = value
    this
  }

  /**
    * Setter for the cardinality check mode to be used.  Available modes are "warn" and "silent".
    * In "warn" mode, an exception will be thrown if the cardinality for a categorical column is above the threshold.
    * In "silent" mode, the field will be ignored from processing and will not be included in the feature vector.
    *
    * @note Default: "silent"
    * @param value String: either "warn" or "silent"
    * @throws IllegalArgumentException if the mode supplied is not either "warn" or "silent"
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[IllegalArgumentException])
  def setFillConfigCardinalityCheckMode(value: String): this.type = {
    validateMembership(
      value,
      allowableCategoricalFilterModes,
      "fillConfigCardinalityCheckMode"
    )
    _instanceConfig.featureEngineeringConfig.cardinalityCheckMode = value
    this
  }

  /**
    * Setter for defining the precision for calculating the model type as per the label column
    * @note setting this value to zero (0) for a large regression problem will incur a long processing time and
    *       an expensive shuffle.
    * @param value Double: Precision accuracy for approximate distinct calculation.
    * @throws java.lang.AssertionError If the value is outside of the allowable range of {0, 1}
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[AssertionError])
  def setFillConfigFilterPrecision(value: Double): this.type = {
    require(
      value >= 0,
      s"Filter Precision for NA Fill must be greater than or equal to 0."
    )
    require(
      value <= 1,
      s"Filter Precision for NA Fill must be less than or equal to 1."
    )
    _instanceConfig.featureEngineeringConfig.filterPrecision = value
    this
  }

  /**
    * Setter for providing a map of [Column Name -> String Fill Value] for manual by-column overrides.  Any non-specified
    * fields in this map will utilize the "auto" statistics-based fill paradigm to calculate and fill any NA values
    * in non-numeric columns.
    * @note if naFillMode is specified as using Map Fill modes, this setter or the numeric na fill map MUST be set.
    * @note If fields are specified in here that are not part of the DataFrame's schema, an exception will be thrown.
    * @param value Map[String, String]: Column Name as String -> Fill Value as String
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  def setFillConfigCategoricalNAFillMap(
    value: Map[String, String]
  ): this.type = {
    _instanceConfig.featureEngineeringConfig.categoricalNAFillMap = value
    this
  }

  /**
    * Setter for providing a map of [Column Name -> AnyVal Fill Value] (must be numeric). Any non-specified
    * fields in this map will utilize the "auto" statistics-based fill paradigm to calculate and fill any NA values
    * in numeric columns.
    * @note if naFillMode is specified as using Map Fill modes, this setter or the categorical na fill map MUST be set.
    * @note If fields are specified in here that are not part of the DataFrame's schema, an exception will be thrown.
    * @param value Map[String, AnyVal]: Column Name as String -> Fill Numeric Type Value
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  def setFillConfigNumericNAFillMap(value: Map[String, AnyVal]): this.type = {
    _instanceConfig.featureEngineeringConfig.numericNAFillMap = value
    this
  }

  /**
    * Setter for providing a 'blanket override' value (fill all found categorical columns' missing values with this
    * specified value).
    * @param value String: A value to fill all categorical na values in the DataFrame with.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  def setFillConfigCharacterNABlanketFillValue(value: String): this.type = {
    _instanceConfig.featureEngineeringConfig.characterNABlanketFillValue = value
    this
  }

  /**
    * Setter for providing a 'blanket override'  value (fill all found numeric columns' missing values with this
    * specified value)
    * @param value Double: A value to fill all numeric na value in the DataFrame with.
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  def setFillConfigNumericNABlanketFillValue(value: Double): this.type = {
    _instanceConfig.featureEngineeringConfig.numericNABlanketFillValue = value
    this
  }

  /**
    * Mode for na fill<br>
    *                Available modes: <br>
    *                  <i>auto</i> : Stats-based na fill for fields.  Usage of .setNumericFillStat and
    *                  .setCharacterFillStat will inform the type of statistics that will be used to fill.<br>
    *                  <i>mapFill</i> : Custom by-column overrides to 'blanket fill' na values on a per-column
    *                  basis.  The categorical (string) fields are set via .setCategoricalNAFillMap while the
    *                  numeric fields are set via .setNumericNAFillMap.<br>
    *                  <i>blanketFillAll</i> : Fills all fields based on the values specified by
    *                  .setCharacterNABlanketFillValue and .setNumericNABlanketFillValue.  All NA's for the
    *                  appropriate types will be filled in accordingly throughout all columns.<br>
    *                  <i>blanketFillCharOnly</i> Will use statistics to fill in numeric fields, but will replace
    *                  all categorical character fields na values with a blanket fill value. <br>
    *                  <i>blanketFillNumOnly</i> Will use statistics to fill in character fields, but will replace
    *                  all numeric fields na values with a blanket value.
    * @throws IllegalArgumentException if the mods specified is not supported.
    * @param value String: Mode for NA Fill
    * @since 0.5.2
    * @author Ben Wilson, Databricks
    */
  @throws(classOf[IllegalArgumentException])
  def setFillConfigNAFillMode(value: String): this.type = {
    validateMembership(value, allowableNAFillModes, "fillConfigNAFillMode")
    _instanceConfig.featureEngineeringConfig.naFillMode = value
    this
  }

  /**
    * Algorithm Config
    */
  def setStringBoundaries(value: Map[String, List[String]]): this.type = {
    validateStringBoundariesKeys(modelType, value)
    _instanceConfig.algorithmConfig.stringBoundaries = value
    this
  }

  def setNumericBoundaries(value: Map[String, (Double, Double)]): this.type = {
    validateNumericBoundariesValues(value)
    validateNumericBoundariesKeys(modelType, value)
    _instanceConfig.algorithmConfig.numericBoundaries = value
    this
  }

  /**
    * Tuner Config
    */
  def setTunerAutoStoppingScore(value: Double): this.type = {
    _instanceConfig.tunerConfig.tunerAutoStoppingScore = value
    this
  }

  def setTunerParallelism(value: Int): this.type = {
    if (value > 30)
      println(
        "WARNING - Setting Tuner Parallelism greater than 30 could put excessive stress on the " +
          "Driver.  Ensure driver is monitored for stability."
      )
    _instanceConfig.tunerConfig.tunerParallelism = value
    this
  }

  def setTunerKFold(value: Int): this.type = {
    if (value < 5)
      println(
        "WARNING - Setting KFold < 5 may result in a poorly generalized tuning run due to " +
          "over-fitting within a particular train/test split."
      )
    _instanceConfig.tunerConfig.tunerKFold = value
    this
  }

  def setTunerTrainPortion(value: Double): this.type = {
    require(
      value > 0.0 & value < 1.0,
      s"TunerTrainPortion must be within the range of 0.0 to 1.0."
    )
    if (value < 0.5)
      println(
        s"WARNING - setting TunerTrainPortion below 0.5 may result in a poorly fit model.  Best" +
          s" practices guidance typically adheres to a 0.7 or 0.8 test/train ratio."
      )
    _instanceConfig.tunerConfig.tunerTrainPortion = value
    this
  }

  def setTunerTrainSplitMethod(value: String): this.type = {
    validateMembership(
      value,
      allowableTrainSplitMethods,
      "TunerTrainSplitMethod"
    )
    _instanceConfig.tunerConfig.tunerTrainSplitMethod = value
    this
  }

  /**
    * Setter - for setting the name of the Synthetic column name
    *
    * @param value String: A column name that is uniquely not part of the main DataFrame
    * @since 0.5.1
    * @author Ben Wilson
    */
  def setTunerKSampleSyntheticCol(value: String): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleSyntheticCol = value
    this
  }

  /**
    * Setter for specifying the number of K-Groups to generate in the KMeans model
    *
    * @param value Int: number of k groups to generate
    * @return this
    */
  def setTunerKSampleKGroups(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleKGroups = value
    this
  }

  /**
    * Setter for specifying the maximum number of iterations for the KMeans model to go through to converge
    *
    * @param value Int: Maximum limit on iterations
    * @return this
    */
  def setTunerKSampleKMeansMaxIter(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleKMeansMaxIter = value
    this
  }

  /**
    * Setter for Setting the tolerance for KMeans (must be >0)
    *
    * @param value The tolerance value setting for KMeans
    * @see reference: [[http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.clustering.KMeans]]
    *      for further details.
    * @return this
    * @throws IllegalArgumentException() if a value less than 0 is entered
    */
  @throws(classOf[IllegalArgumentException])
  def setTunerKSampleKMeansTolerance(value: Double): this.type = {
    require(
      value > 0,
      s"KMeans tolerance value ${value.toString} is out of range.  Must be > 0."
    )
    _instanceConfig.tunerConfig.tunerKSampleKMeansTolerance = value
    this
  }

  /**
    * Setter for which distance measurement to use to calculate the nearness of vectors to a centroid
    *
    * @param value String: Options -> "euclidean" or "cosine" Default: "euclidean"
    * @return this
    * @throws IllegalArgumentException() if an invalid value is entered
    */
  @throws(classOf[IllegalArgumentException])
  def setTunerKSampleKMeansDistanceMeasurement(value: String): this.type = {
    validateMembership(
      value,
      allowableKMeansDistanceMeasurements,
      "tunerKSampleKMeansDistanceMeasurement"
    )
    _instanceConfig.tunerConfig.tunerKSampleKMeansDistanceMeasurement = value
    this
  }

  /**
    * Setter for a KMeans seed for the clustering algorithm
    *
    * @param value Long: Seed value
    * @return this
    */
  def setTunerKSampleKMeansSeed(value: Long): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleKMeansSeed = value
    this
  }

  /**
    * Setter for the internal KMeans column for cluster membership attribution
    *
    * @param value String: column name for internal algorithm column for group membership
    * @return this
    */
  def setTunerKSampleKMeansPredictionCol(value: String): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleKMeansPredictionCol = value
    this
  }

  /**
    * Setter for Configuring the number of Hash Tables to use for MinHashLSH
    *
    * @param value Int: Count of hash tables to use
    * @see [[http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.MinHashLSH]]
    *     for more information
    * @return this
    */
  def setTunerKSampleLSHHashTables(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleLSHHashTables = value
    this
  }

  def setTunerKSampleLSHSeed(value: Long): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleLSHSeed = value
    this
  }

  /**
    * Setter for the internal LSH output hash information column
    *
    * @param value String: column name for the internal MinHashLSH Model transformation value
    * @return this
    */
  def setTunerKSampleLSHOutputCol(value: String): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleLSHOutputCol = value
    this
  }

  /**
    * Setter for how many vectors to find in adjacency to the centroid for generation of synthetic data
    *
    * @note the higher the value set here, the higher the variance in synthetic data generation
    * @param value Int: Number of vectors to find nearest each centroid within the class
    * @return this
    */
  def setTunerKSampleQuorumCount(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleQuorumCount = value
    this
  }

  /**
    * Setter for minimum threshold for vector indexes to mutate within the feature vector.
    *
    * @note In vectorMutationMethod "fixed" this sets the fixed count of how many vector positions to mutate.
    *       In vectorMutationMethod "random" this sets the lower threshold for 'at least this many indexes will
    *       be mutated'
    * @param value The minimum (or fixed) number of indexes to mutate.
    * @return this
    */
  def setTunerKSampleMinimumVectorCountToMutate(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleMinimumVectorCountToMutate = value
    this
  }

  /**
    * Setter for the Vector Mutation Method
    *
    * @note Options:
    *       "fixed" - will use the value of minimumVectorCountToMutate to select random indexes of this number of indexes.
    *       "random" - will use this number as a lower bound on a random selection of indexes between this and the vector length.
    *       "all" - will mutate all of the vectors.
    * @param value String - the mode to use.
    * @return this
    * @throws IllegalArgumentException() if the mode is not supported.
    */
  @throws(classOf[IllegalArgumentException])
  def setTunerKSampleVectorMutationMethod(value: String): this.type = {
    validateMembership(
      value,
      allowableVectorMutationMethods,
      "tunerKSampleVectorMutationMethod"
    )
    _instanceConfig.tunerConfig.tunerKSampleVectorMutationMethod = value
    this
  }

  /**
    * Setter for the Mutation Mode of the feature vector individual values
    *
    * @note Options:
    *       "weighted" - uses weighted averaging to scale the euclidean distance between the centroid vector and mutation candidate vectors
    *       "random" - randomly selects a position on the euclidean vector between the centroid vector and the candidate mutation vectors
    *       "ratio" - uses a ratio between the values of the centroid vector and the mutation vector    *
    * @param value String: the mode to use.
    * @return this
    * @throws IllegalArgumentException() if the mode is not supported.
    */
  @throws(classOf[IllegalArgumentException])
  def setTunerKSampleMutationMode(value: String): this.type = {
    validateMembership(
      value,
      allowableMutationModes,
      "tunerKSampleMutationMode"
    )
    _instanceConfig.tunerConfig.tunerKSampleMutationMode = value
    this
  }

  /**
    * Setter for specifying the mutation magnitude for the modes 'weighted' and 'ratio' in mutationMode
    *
    * @param value Double: value between 0 and 1 for mutation magnitude adjustment.
    * @note the higher this value, the closer to the centroid vector vs. the candidate mutation vector the synthetic row data will be.
    * @return this
    * @throws IllegalArgumentException() if the value specified is outside of the range (0, 1)
    */
  @throws(classOf[IllegalArgumentException])
  def setTunerKSampleMutationValue(value: Double): this.type = {
    require(
      value > 0 & value < 1,
      s"Mutation Value must be between 0 and 1. Value $value is not permitted."
    )
    _instanceConfig.tunerConfig.tunerKSampleMutationValue = value
    this
  }

  /**
    * Setter - for determining the label balance approach mode.
    *
    * @note Available modes: <br>
    *         <i>'match'</i>: Will match all smaller class counts to largest class count.  [WARNING] - May significantly increase memory pressure!<br>
    *         <i>'percentage'</i> Will adjust smaller classes to a percentage value of the largest class count.
    *         <i>'target'</i> Will increase smaller class counts to a fixed numeric target of rows.
    * @param value String: one of: 'match', 'percentage' or 'target'
    * @note Default: "percentage"
    * @since 0.5.1
    * @author Ben Wilson
    * @throws IllegalArgumentException if the provided mode is not supported.
    */
  @throws(classOf[IllegalArgumentException])
  def setTunerKSampleLabelBalanceMode(value: String): this.type = {
    validateMembership(
      value,
      allowableLabelBalanceModes,
      "tunerKSampleLabelBalanceMode"
    )
    _instanceConfig.tunerConfig.tunerKSampleLabelBalanceMode = value
    this
  }

  /**
    * Setter - for overriding the cardinality threshold exception threshold.  [WARNING] increasing this value on
    * a sufficiently large data set could incur, during runtime, excessive memory and cpu pressure on the cluster.
    *
    * @param value Int: the limit above which an exception will be thrown for a classification problem wherein the
    *              label distinct count is too large to successfully generate synthetic data.
    * @note Default: 20
    * @since 0.5.1
    * @author Ben Wilson
    */
  def setTunerKSampleCardinalityThreshold(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleCardinalityThreshold = value
    this
  }

  /**
    * Setter - for specifying the percentage ratio for the mode 'percentage' in setLabelBalanceMode()
    *
    * @param value Double: A fractional double in the range of 0.0 to 1.0.
    * @note Setting this value to 1.0 is equivalent to setting the label balance mode to 'match'
    * @note Default: 0.2
    * @since 0.5.1
    * @author Ben Wilson
    * @throws UnsupportedOperationException() if the provided value is outside of the range of 0.0 -> 1.0
    */
  @throws(classOf[UnsupportedOperationException])
  def setTunerKSampleNumericRatio(value: Double): this.type = {
    require(
      value <= 1.0 & value > 0.0,
      s"Invalid Numeric Ratio entered!  Must be between 0 and 1." +
        s"${value.toString} is not valid."
    )
    _instanceConfig.tunerConfig.tunerKSampleNumericRatio = value
    this
  }

  /**
    * Setter - for specifying the target row count to generate for 'target' mode in setLabelBalanceMode()
    *
    * @param value Int: The desired final number of rows per minority class label
    * @note [WARNING] Setting this value to too high of a number will greatly increase runtime and memory pressure.
    * @since 0.5.1
    * @author Ben Wilson
    */
  def setTunerKSampleNumericTarget(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerKSampleNumericTarget = value
    this
  }

  def setTunerTrainSplitChronologicalColumn(value: String): this.type = {
    _instanceConfig.tunerConfig.tunerTrainSplitChronologicalColumn = value
    if (value.length > 0) {
      val updatedFieldsToIgnore = genericConfig.fieldsToIgnoreInVector ++ Array(
        value
      )
      genericConfig.fieldsToIgnoreInVector = updatedFieldsToIgnore
    }
    this
  }

  def setTunerTrainSplitChronologicalRandomPercentage(
    value: Double
  ): this.type = {
    if (value > 10)
      println(
        "[WARNING] TunerTrainSplitChronologicalRandomPercentage setting this value above 10 " +
          "percent will cause significant per-run train/test skew and variability in row counts during training.  " +
          "Use higher values only if this is desired."
      )
    _instanceConfig.tunerConfig.tunerTrainSplitChronologicalRandomPercentage =
      value
    this
  }

  def setTunerSeed(value: Long): this.type = {
    _instanceConfig.tunerConfig.tunerSeed = value
    this
  }

  def setTunerFirstGenerationGenePool(value: Int): this.type = {
    if (value < 10)
      println(
        "[WARNING] TunerFirstGenerationGenePool values of less than 10 may not find global minima" +
          "for hyperparameters.  Consider setting the value > 30 for best performance."
      )
    _instanceConfig.tunerConfig.tunerFirstGenerationGenePool = value
    this
  }

  def setTunerNumberOfGenerations(value: Int): this.type = {
    if (value < 3)
      println(
        "[WARNING] TunerNumberOfGenerations set below 3 may not explore hyperparameter feature " +
          "space effectively to arrive at a global minima."
      )
    if (value > 20)
      println(
        "[WARNING] TunerNumberOfGenerations set above 20 will take a long time to run.  Evaluate" +
          "whether first generation gene pool count and numer of mutations per generation should be adjusted higher" +
          "instead."
      )
    _instanceConfig.tunerConfig.tunerNumberOfGenerations = value
    this
  }

  def setTunerNumberOfParentsToRetain(value: Int): this.type = {
    require(
      value > 0,
      s"TunerNumberOfParentsToRetain must be > 0. $value is outside of bounds."
    )
    _instanceConfig.tunerConfig.tunerNumberOfParentsToRetain = value
    this
  }

  def setTunerNumberOfMutationsPerGeneration(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerNumberOfMutationsPerGeneration = value
    this
  }

  def setTunerGeneticMixing(value: Double): this.type = {
    zeroToOneValidation(value, "TunerGeneticMixing")
    if (value > 0.9)
      println(
        s"[WARNING] Setting TunerGeneticMixing to a value greater than 0.9 will not effectively" +
          s"explore the hyperparameter feature space.  Use such settings only for fine-tuning around a pre-calculated " +
          s"global minima."
      )
    _instanceConfig.tunerConfig.tunerGeneticMixing = value
    this
  }

  def setTunerGenerationalMutationStrategy(value: String): this.type = {
    validateMembership(
      value,
      allowableMutationStrategies,
      "TunerGenerationalMutationStrategy"
    )
    _instanceConfig.tunerConfig.tunerGenerationalMutationStrategy = value
    this
  }

  def setTunerFixedMutationValue(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerFixedMutationValue = value
    this
  }

  def setTunerMutationMagnitudeMode(value: String): this.type = {
    validateMembership(
      value,
      allowableMutationMagnitudeMode,
      "TunerMutationMagnitudeMode"
    )
    _instanceConfig.tunerConfig.tunerMutationMagnitudeMode = value
    this
  }

  def setTunerEvolutionStrategy(value: String): this.type = {
    validateMembership(
      value,
      allowableEvolutionStrategies,
      "TunerEvolutionStrategy"
    )
    _instanceConfig.tunerConfig.tunerEvolutionStrategy = value
    this
  }

  /**
    * Setter for defining the secondary stopping criteria for continuous training mode ( number of consistentlt
    * not-improving runs to terminate the learning algorithm due to diminishing returns.
    * @param value Negative Integer (an improvement to a priori will reset the counter and subsequent non-improvements
    *              will decrement a mutable counter.  If the counter hits this limit specified in value, the continuous
    *              mode algorithm will stop).
    * @author Ben Wilson, Databricks
    * @since 0.6.0
    * @throws IllegalArgumentException if the value is positive.
    */
  @throws(classOf[IllegalArgumentException])
  def setTunerContinuousEvolutionImprovementThreshold(value: Int): this.type = {
    require(
      value < 0,
      s"ContinuousEvolutionImprovementThreshold must be less than zero.  It is " +
        s"recommended to set this value to less than -4."
    )
    _instanceConfig.tunerConfig.tunerContinuousEvolutionImprovementThreshold =
      value
    this
  }

  /**
    * Setter for selecting the type of Regressor to use for the within-epoch generation MBO of candidates
    * @param value String - one of "XGBoost", "LinearRegression" or "RandomForest"
    * @author Ben Wilson, Databricks
    * @since 0.6.0
    * @throws IllegalArgumentException if the value is not supported
    */
  @throws(classOf[IllegalArgumentException])
  def setTunerGeneticMBORegressorType(value: String): this.type = {
    validateMembership(
      value,
      allowableGeneticMBORegressorTypes,
      "GeneticMBORegressorType"
    )
    _instanceConfig.tunerConfig.tunerGeneticMBORegressorType = value
    this
  }

  /**
    * Setter for defining the factor to be applied to the candidate listing of hyperparameters to generate through
    * mutation for each generation other than the initial and post-modeling optimization phases.  The larger this
    * value (default: 10), the more potential space can be searched.  There is not a large performance hit to this,
    * and as such, values in excess of 100 are viable.
    * @param value Int - a factor to multiply the numberOfMutationsPerGeneration by to generate a count of potential
    *              candidates.
    * @author Ben Wilson, Databricks
    * @since 0.6.0
    * @throws IllegalArgumentException if the value is not greater than zero.
    */
  @throws(classOf[IllegalArgumentException])
  def setTunerGeneticMBOCandidateFactor(value: Int): this.type = {
    require(value > 0, s"GeneticMBOCandidateFactor must be greater than zero.")
    _instanceConfig.tunerConfig.tunerGeneticMBOCandidateFactor = value
    this
  }

  def setTunerContinuousEvolutionMaxIterations(value: Int): this.type = {
    if (value > 500)
      println(
        s"[WARNING] Setting this value higher increases runtime by O(n/parallelism) amount.  " +
          s"Values higher than 500 may take an unacceptably long time to run. "
      )
    _instanceConfig.tunerConfig.tunerContinuousEvolutionMaxIterations = value
    this
  }

  def setTunerContinuousEvolutionStoppingScore(value: Double): this.type = {
    zeroToOneValidation(value, "TunerContinuuousEvolutionStoppingScore")
    _instanceConfig.tunerConfig.tunerContinuousEvolutionStoppingScore = value
    this
  }

  def setTunerContinuousEvolutionParallelism(value: Int): this.type = {
    if (value > 10)
      println(
        "[WARNING] Setting value of TunerContinuousEvolutionParallelism greater than 10 may have" +
          "unintended side-effects of a longer convergence time due to async Futures that have not returned results" +
          "by the time that the next iteration is initiated.  Recommended settings are in the range of [4:8] for " +
          "continuous mode."
      )
    _instanceConfig.tunerConfig.tunerContinuousEvolutionParallelism = value
    this
  }

  def setTunerContinuousEvolutionMutationAggressiveness(
    value: Int
  ): this.type = {
    _instanceConfig.tunerConfig.tunerContinuousEvolutionMutationAggressiveness =
      value
    this
  }

  def setTunerContinuousEvolutionGeneticMixing(value: Double): this.type = {
    zeroToOneValidation(value, "TunerContinuousEvolutionGeneticMixing")
    if (value > 0.9)
      println(
        s"[WARNING] Setting TunerContinuousEvolutionGeneticMixing to a value greater than 0.9 " +
          s"will not effectively explore the hyperparameter feature space.  Use such settings only for fine-tuning " +
          s"around a pre-calculated global minima."
      )
    _instanceConfig.tunerConfig.tunerContinuousEvolutionGeneticMixing = value
    this
  }

  def setTunerContinuousEvolutionRollingImprovementCount(
    value: Int
  ): this.type = {
    _instanceConfig.tunerConfig.tunerContinuousEvolutionRollingImprovingCount =
      value
    this
  }

  //TODO: per model validation of keys?
  def setTunerModelSeed(value: Map[String, Any]): this.type = {
    _instanceConfig.tunerConfig.tunerModelSeed = value
    this
  }

  def setTunerHyperSpaceInferenceOn(): this.type = {
    _instanceConfig.tunerConfig.tunerHyperSpaceInference = true
    this
  }

  def setTunerHyperSpaceInferenceOff(): this.type = {
    _instanceConfig.tunerConfig.tunerHyperSpaceInference = false
    this
  }

  def setTunerHyperSpaceInferenceFlag(value: Boolean): this.type = {
    _instanceConfig.tunerConfig.tunerHyperSpaceInference = value
    this
  }

  def setTunerHyperSpaceInferenceCount(value: Int): this.type = {
    if (value > 500000)
      println(
        "[WARNING] Setting TunerHyperSpaceInferenceCount above 500,000 will put stress on the " +
          "driver for generating so many leaves."
      )
    if (value > 1000000)
      throw new UnsupportedOperationException(
        s"Setting TunerHyperSpaceInferenceCount above " +
          s"1,000,000 is not supported due to runtime considerations.  $value is too large of a value."
      )
    _instanceConfig.tunerConfig.tunerHyperSpaceInferenceCount = value
    this
  }

  def setTunerHyperSpaceModelCount(value: Int): this.type = {
    if (value > 50)
      println(
        "[WARNING] TunerHyperSpaceModelCount values set excessively high will incur long runtime" +
          "costs after the conclusion of Genetic Tuner running.  Gains are diminishing after a value of 20."
      )
    _instanceConfig.tunerConfig.tunerHyperSpaceModelCount = value
    this
  }

  def setTunerHyperSpaceModelType(value: String): this.type = {
    validateMembership(
      value,
      allowableHyperSpaceModelTypes,
      "TunerHyperSpaceModelType"
    )
    _instanceConfig.tunerConfig.tunerHyperSpaceModelType = value
    this
  }

  def setTunerInitialGenerationMode(value: String): this.type = {
    validateMembership(
      value,
      allowableInitialGenerationModes,
      "TunerInitialGenerationMode"
    )
    _instanceConfig.tunerConfig.tunerInitialGenerationMode = value
    this
  }

  def setTunerInitialGenerationPermutationCount(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerInitialGenerationPermutationCount = value
    this
  }

  def setTunerInitialGenerationIndexMixingMode(value: String): this.type = {
    validateMembership(
      value,
      allowableInitialGenerationIndexMixingModes,
      "TunerInitialGenerationIndexMixingMode"
    )
    _instanceConfig.tunerConfig.tunerInitialGenerationIndexMixingMode = value
    this
  }

  def setTunerInitialGenerationArraySeed(value: Long): this.type = {
    _instanceConfig.tunerConfig.tunerInitialGenerationArraySeed = value
    this
  }

  def setTunerOutputDfRepartitionScaleFactor(value: Int): this.type = {
    _instanceConfig.tunerConfig.tunerOutputDfRepartitionScaleFactor = value
    this
  }

  /**
    * MLFlow Logging Config
    */
  def setMlFlowLoggingOn(): this.type = {
    _instanceConfig.loggingConfig.mlFlowLoggingFlag = true
    this
  }

  def setMlFlowLoggingOff(): this.type = {
    _instanceConfig.loggingConfig.mlFlowLoggingFlag = false
    this
  }

  def setMlFlowLoggingFlag(value: Boolean): this.type = {
    _instanceConfig.loggingConfig.mlFlowLoggingFlag = value
    this
  }

  def setMlFlowLogArtifactsOn(): this.type = {
    _instanceConfig.loggingConfig.mlFlowLogArtifactsFlag = true
    this
  }

  def setMlFlowLogArtifactsOff(): this.type = {
    _instanceConfig.loggingConfig.mlFlowLogArtifactsFlag = false
    this
  }

  def setMlFlowLogArtifactsFlag(value: Boolean): this.type = {
    _instanceConfig.loggingConfig.mlFlowLogArtifactsFlag = value
    this
  }

  //TODO: Add path validation here!!
  def setMlFlowTrackingURI(value: String): this.type = {
    _instanceConfig.loggingConfig.mlFlowTrackingURI = value
    this
  }

  def setMlFlowExperimentName(value: String): this.type = {
    _instanceConfig.loggingConfig.mlFlowExperimentName = value
    this
  }

  def setMlFlowAPIToken(value: String): this.type = {
    _instanceConfig.loggingConfig.mlFlowAPIToken = value
    this
  }

  def setMlFlowModelSaveDirectory(value: String): this.type = {
    _instanceConfig.loggingConfig.mlFlowModelSaveDirectory = value
    this
  }

  def setMlFlowLoggingMode(value: String): this.type = {
    validateMembership(value, allowableMlFlowLoggingModes, "MlFlowLoggingMode")
    _instanceConfig.loggingConfig.mlFlowLoggingMode = value
    this
  }

  def setMlFlowBestSuffix(value: String): this.type = {
    _instanceConfig.loggingConfig.mlFlowBestSuffix = value
    this
  }

  def setInferenceConfigSaveLocation(value: String): this.type = {
    _instanceConfig.loggingConfig.inferenceConfigSaveLocation = value
    this
  }

  /**
    * Setter<br>
    *   Allows for setting a series of custom mlflow logging tags to an experiment run (universal across all
    *   iterations and models of the run) to be logged in mlflow as a custom tag key value pair
    *
    * @param value Array of Map[String -> AnyVal]
    * @note The mapped values can be of types: Double, Float, Long, Int, Short, Byte, Boolean, or String
    */
  def setMlFlowCustomRunTags(value: Map[String, AnyVal]): this.type = {

    val parsedValue =
      value.map { case (k, v) => k -> v.asInstanceOf[String] }

    _instanceConfig.loggingConfig.mlFlowCustomRunTags = parsedValue
    this
  }

  /**
    * Getters
    */
  def getInstanceConfig: InstanceConfig = _instanceConfig

  def generateMainConfig: MainConfig =
    ConfigurationGenerator.generateMainConfig(_instanceConfig)

  def generateFeatureImportanceConfig: MainConfig =
    ConfigurationGenerator.generateMainConfig(_instanceConfig)

  def generateTreeSplitConfig: MainConfig =
    ConfigurationGenerator.generateMainConfig(_instanceConfig)

}

object ConfigurationGenerator extends ConfigurationDefaults {

  import PredictionType._

  def apply(modelFamily: String,
            predictionType: String,
            genericConfig: GenericConfig): ConfigurationGenerator =
    new ConfigurationGenerator(modelFamily, predictionType, genericConfig)

  /**
    *
    * @param modelFamily
    * @param predictionType
    * @return
    */
  def generateDefaultConfig(modelFamily: String,
                            predictionType: String): InstanceConfig = {

    predictionTypeEvaluator(predictionType) match {
      case Regressor =>
        new ConfigurationGenerator(
          modelFamily,
          predictionType,
          GenericConfigGenerator.generateDefaultRegressorConfig
        ).getInstanceConfig
      case Classifier =>
        new ConfigurationGenerator(
          modelFamily,
          predictionType,
          GenericConfigGenerator.generateDefaultClassifierConfig
        ).getInstanceConfig
    }

  }

  private def standardizeModelFamilyStrings(value: String): String = {
    value.toLowerCase match {
      case "randomforest"       => "RandomForest"
      case "gbt"                => "GBT"
      case "linearregression"   => "LinearRegression"
      case "logisticregression" => "LogisticRegression"
      case "mlpc"               => "MLPC"
      case "svm"                => "SVM"
      case "trees"              => "Trees"
      case "xgboost"            => "XGBoost"
      case "gbmbinary"          => "gbmBinary"
      case "gbmmulti"           => "gbmMulti"
      case "gbmmultiova"        => "gbmMultiOVA"
      case "gbmhuber"           => "gbmHuber"
      case "gbmfair"            => "gbmFair"
      case "gbmlasso"           => "gbmLasso"
      case "gbmridge"           => "gbmRidge"
      case "gbmpoisson"         => "gbmPoisson"
      case "gbmquantile"        => "gbmQuantile"
      case "gbmmape"            => "gbmMape"
      case "gbmtweedie"         => "gbmTweedie"
      case "gbmgamma"           => "gbmGamma"
      case _ =>
        throw new IllegalArgumentException(
          s"standardizeModelFamilyStrings does not have a supported" +
            s"type of: $value"
        )
    }
  }

  /**
    *
    * @param config
    * @return
    */
  def generateMainConfig(config: InstanceConfig): MainConfig = {
    MainConfig(
      modelFamily = standardizeModelFamilyStrings(config.modelFamily),
      labelCol = config.genericConfig.labelCol,
      featuresCol = config.genericConfig.featuresCol,
      naFillFlag = config.switchConfig.naFillFlag,
      varianceFilterFlag = config.switchConfig.varianceFilterFlag,
      outlierFilterFlag = config.switchConfig.outlierFilterFlag,
      pearsonFilteringFlag = config.switchConfig.pearsonFilterFlag,
      covarianceFilteringFlag = config.switchConfig.covarianceFilterFlag,
      oneHotEncodeFlag = config.switchConfig.oneHotEncodeFlag,
      scalingFlag = config.switchConfig.scalingFlag,
      featureInteractionFlag = config.switchConfig.featureInteractionFlag,
      dataPrepCachingFlag = config.switchConfig.dataPrepCachingFlag,
      dataPrepParallelism = config.featureEngineeringConfig.dataPrepParallelism,
      autoStoppingFlag = config.switchConfig.autoStoppingFlag,
      autoStoppingScore = config.tunerConfig.tunerAutoStoppingScore,
      featureImportanceCutoffType =
        config.featureEngineeringConfig.featureImportanceCutoffType,
      featureImportanceCutoffValue =
        config.featureEngineeringConfig.featureImportanceCutoffValue,
      dateTimeConversionType = config.genericConfig.dateTimeConversionType,
      fieldsToIgnoreInVector = config.genericConfig.fieldsToIgnoreInVector,
      numericBoundaries = config.algorithmConfig.numericBoundaries,
      stringBoundaries = config.algorithmConfig.stringBoundaries,
      scoringMetric = config.genericConfig.scoringMetric,
      scoringOptimizationStrategy =
        config.genericConfig.scoringOptimizationStrategy,
      fillConfig = FillConfig(
        numericFillStat = config.featureEngineeringConfig.numericFillStat,
        characterFillStat = config.featureEngineeringConfig.characterFillStat,
        modelSelectionDistinctThreshold =
          config.featureEngineeringConfig.modelSelectionDistinctThreshold,
        cardinalitySwitch = config.featureEngineeringConfig.cardinalitySwitch,
        cardinalityType = config.featureEngineeringConfig.cardinalityType,
        cardinalityLimit = config.featureEngineeringConfig.cardinalityLimit,
        cardinalityPrecision =
          config.featureEngineeringConfig.cardinalityPrecision,
        cardinalityCheckMode =
          config.featureEngineeringConfig.cardinalityCheckMode,
        filterPrecision = config.featureEngineeringConfig.filterPrecision,
        categoricalNAFillMap =
          config.featureEngineeringConfig.categoricalNAFillMap,
        numericNAFillMap = config.featureEngineeringConfig.numericNAFillMap,
        characterNABlanketFillValue =
          config.featureEngineeringConfig.characterNABlanketFillValue,
        numericNABlanketFillValue =
          config.featureEngineeringConfig.numericNABlanketFillValue,
        naFillMode = config.featureEngineeringConfig.naFillMode
      ),
      outlierConfig = OutlierConfig(
        filterBounds = config.featureEngineeringConfig.outlierFilterBounds,
        lowerFilterNTile =
          config.featureEngineeringConfig.outlierLowerFilterNTile,
        upperFilterNTile =
          config.featureEngineeringConfig.outlierUpperFilterNTile,
        filterPrecision = config.featureEngineeringConfig.outlierFilterPrecision,
        continuousDataThreshold =
          config.featureEngineeringConfig.outlierContinuousDataThreshold,
        fieldsToIgnore = config.featureEngineeringConfig.outlierFieldsToIgnore
      ),
      pearsonConfig = PearsonConfig(
        filterStatistic = config.featureEngineeringConfig.pearsonFilterStatistic,
        filterDirection = config.featureEngineeringConfig.pearsonFilterDirection,
        filterManualValue =
          config.featureEngineeringConfig.pearsonFilterManualValue,
        filterMode = config.featureEngineeringConfig.pearsonFilterMode,
        autoFilterNTile = config.featureEngineeringConfig.pearsonAutoFilterNTile
      ),
      covarianceConfig = CovarianceConfig(
        correlationCutoffLow =
          config.featureEngineeringConfig.covarianceCorrelationCutoffLow,
        correlationCutoffHigh =
          config.featureEngineeringConfig.covarianceCorrelationCutoffHigh
      ),
      featureInteractionConfig = FeatureInteractionConfig(
        retentionMode =
          config.featureEngineeringConfig.featureInteractionRetentionMode,
        continuousDiscretizerBucketCount =
          config.featureEngineeringConfig.featureInteractionContinuousDiscretizerBucketCount,
        parallelism =
          config.featureEngineeringConfig.featureInteractionParallelism,
        targetInteractionPercentage =
          config.featureEngineeringConfig.featureInteractionTargetInteractionPercentage
      ),
      scalingConfig = ScalingConfig(
        scalerType = config.featureEngineeringConfig.scalingType,
        scalerMin = config.featureEngineeringConfig.scalingMin,
        scalerMax = config.featureEngineeringConfig.scalingMax,
        standardScalerMeanFlag =
          config.featureEngineeringConfig.scalingStandardMeanFlag,
        standardScalerStdDevFlag =
          config.featureEngineeringConfig.scalingStdDevFlag,
        pNorm = config.featureEngineeringConfig.scalingPNorm
      ),
      geneticConfig = GeneticConfig(
        parallelism = config.tunerConfig.tunerParallelism,
        kFold = config.tunerConfig.tunerKFold,
        trainPortion = config.tunerConfig.tunerTrainPortion,
        trainSplitMethod = config.tunerConfig.tunerTrainSplitMethod,
        kSampleConfig = KSampleConfig(
          syntheticCol = config.tunerConfig.tunerKSampleSyntheticCol,
          kGroups = config.tunerConfig.tunerKSampleKGroups,
          kMeansMaxIter = config.tunerConfig.tunerKSampleKMeansMaxIter,
          kMeansTolerance = config.tunerConfig.tunerKSampleKMeansTolerance,
          kMeansDistanceMeasurement =
            config.tunerConfig.tunerKSampleKMeansDistanceMeasurement,
          kMeansSeed = config.tunerConfig.tunerKSampleKMeansSeed,
          kMeansPredictionCol =
            config.tunerConfig.tunerKSampleKMeansPredictionCol,
          lshHashTables = config.tunerConfig.tunerKSampleLSHHashTables,
          lshSeed = config.tunerConfig.tunerKSampleLSHSeed,
          lshOutputCol = config.tunerConfig.tunerKSampleLSHOutputCol,
          quorumCount = config.tunerConfig.tunerKSampleQuorumCount,
          minimumVectorCountToMutate =
            config.tunerConfig.tunerKSampleMinimumVectorCountToMutate,
          vectorMutationMethod =
            config.tunerConfig.tunerKSampleVectorMutationMethod,
          mutationMode = config.tunerConfig.tunerKSampleMutationMode,
          mutationValue = config.tunerConfig.tunerKSampleMutationValue,
          labelBalanceMode = config.tunerConfig.tunerKSampleLabelBalanceMode,
          cardinalityThreshold =
            config.tunerConfig.tunerKSampleCardinalityThreshold,
          numericRatio = config.tunerConfig.tunerKSampleNumericRatio,
          numericTarget = config.tunerConfig.tunerKSampleNumericTarget,
          outputDfRepartitionScaleFactor =
            config.tunerConfig.tunerOutputDfRepartitionScaleFactor
        ),
        trainSplitChronologicalColumn =
          config.tunerConfig.tunerTrainSplitChronologicalColumn,
        trainSplitChronologicalRandomPercentage =
          config.tunerConfig.tunerTrainSplitChronologicalRandomPercentage,
        seed = config.tunerConfig.tunerSeed,
        firstGenerationGenePool =
          config.tunerConfig.tunerFirstGenerationGenePool,
        numberOfGenerations = config.tunerConfig.tunerNumberOfGenerations,
        numberOfParentsToRetain =
          config.tunerConfig.tunerNumberOfParentsToRetain,
        numberOfMutationsPerGeneration =
          config.tunerConfig.tunerNumberOfMutationsPerGeneration,
        geneticMixing = config.tunerConfig.tunerGeneticMixing,
        generationalMutationStrategy =
          config.tunerConfig.tunerGenerationalMutationStrategy,
        fixedMutationValue = config.tunerConfig.tunerFixedMutationValue,
        mutationMagnitudeMode = config.tunerConfig.tunerMutationMagnitudeMode,
        evolutionStrategy = config.tunerConfig.tunerEvolutionStrategy,
        geneticMBORegressorType =
          config.tunerConfig.tunerGeneticMBORegressorType,
        geneticMBOCandidateFactor =
          config.tunerConfig.tunerGeneticMBOCandidateFactor,
        continuousEvolutionMaxIterations =
          config.tunerConfig.tunerContinuousEvolutionMaxIterations,
        continuousEvolutionStoppingScore =
          config.tunerConfig.tunerContinuousEvolutionStoppingScore,
        continuousEvolutionImprovementThreshold =
          config.tunerConfig.tunerContinuousEvolutionImprovementThreshold,
        continuousEvolutionParallelism =
          config.tunerConfig.tunerContinuousEvolutionParallelism,
        continuousEvolutionMutationAggressiveness =
          config.tunerConfig.tunerContinuousEvolutionMutationAggressiveness,
        continuousEvolutionGeneticMixing =
          config.tunerConfig.tunerContinuousEvolutionGeneticMixing,
        continuousEvolutionRollingImprovementCount =
          config.tunerConfig.tunerContinuousEvolutionRollingImprovingCount,
        modelSeed = config.tunerConfig.tunerModelSeed,
        hyperSpaceInference = config.tunerConfig.tunerHyperSpaceInference,
        hyperSpaceInferenceCount =
          config.tunerConfig.tunerHyperSpaceInferenceCount,
        hyperSpaceModelType = config.tunerConfig.tunerHyperSpaceModelType,
        hyperSpaceModelCount = config.tunerConfig.tunerHyperSpaceModelCount,
        initialGenerationMode = config.tunerConfig.tunerInitialGenerationMode,
        initialGenerationConfig = FirstGenerationConfig(
          permutationCount =
            config.tunerConfig.tunerInitialGenerationPermutationCount,
          indexMixingMode =
            config.tunerConfig.tunerInitialGenerationIndexMixingMode,
          arraySeed = config.tunerConfig.tunerInitialGenerationArraySeed
        )
      ),
      mlFlowLoggingFlag = config.loggingConfig.mlFlowLoggingFlag,
      mlFlowLogArtifactsFlag = config.loggingConfig.mlFlowLogArtifactsFlag,
      mlFlowConfig = MLFlowConfig(
        mlFlowTrackingURI = config.loggingConfig.mlFlowTrackingURI,
        mlFlowExperimentName = config.loggingConfig.mlFlowExperimentName,
        mlFlowAPIToken = config.loggingConfig.mlFlowAPIToken,
        mlFlowModelSaveDirectory = config.loggingConfig.mlFlowModelSaveDirectory,
        mlFlowLoggingMode = config.loggingConfig.mlFlowLoggingMode,
        mlFlowBestSuffix = config.loggingConfig.mlFlowBestSuffix,
        mlFlowCustomRunTags = config.loggingConfig.mlFlowCustomRunTags
      ),
      inferenceConfigSaveLocation =
        config.loggingConfig.inferenceConfigSaveLocation,
      dataReductionFactor = config.featureEngineeringConfig.dataReductionFactor,
      pipelineDebugFlag = config.switchConfig.pipelineDebugFlag
    )
  }

  /**
    * Helper method for generating the configuration for executing an exploratory FeatureImportance run
    *
    * @param config InstanceConfig Object
    * @return Instance of FeatureImportanceConfig
    * @since 0.5.1
    * @author Ben Wilson
    */
  def generateFeatureImportanceConfig(
    config: InstanceConfig
  ): FeatureImportanceConfig = {

    FeatureImportanceConfig(
      labelCol = config.genericConfig.labelCol,
      featuresCol = config.genericConfig.featuresCol,
      dataPrepParallelism = config.featureEngineeringConfig.dataPrepParallelism,
      numericBoundaries = config.algorithmConfig.numericBoundaries,
      stringBoundaries = config.algorithmConfig.stringBoundaries,
      scoringMetric = config.genericConfig.scoringMetric,
      trainPortion = config.tunerConfig.tunerTrainPortion,
      trainSplitMethod = config.tunerConfig.tunerTrainSplitMethod,
      trainSplitChronologicalColumn =
        config.tunerConfig.tunerTrainSplitChronologicalColumn,
      trainSplitChronlogicalRandomPercentage =
        config.tunerConfig.tunerTrainSplitChronologicalRandomPercentage,
      parallelism = config.tunerConfig.tunerParallelism,
      kFold = config.tunerConfig.tunerKFold,
      seed = config.tunerConfig.tunerSeed,
      scoringOptimizationStrategy =
        config.genericConfig.scoringOptimizationStrategy,
      firstGenerationGenePool = config.tunerConfig.tunerFirstGenerationGenePool,
      numberOfGenerations = config.tunerConfig.tunerNumberOfGenerations,
      numberOfMutationsPerGeneration =
        config.tunerConfig.tunerNumberOfMutationsPerGeneration,
      numberOfParentsToRetain = config.tunerConfig.tunerNumberOfParentsToRetain,
      geneticMixing = config.tunerConfig.tunerGeneticMixing,
      generationalMutationStrategy =
        config.tunerConfig.tunerGenerationalMutationStrategy,
      mutationMagnitudeMode = config.tunerConfig.tunerMutationMagnitudeMode,
      fixedMutationValue = config.tunerConfig.tunerFixedMutationValue,
      autoStoppingScore = config.tunerConfig.tunerAutoStoppingScore,
      autoStoppingFlag = config.switchConfig.autoStoppingFlag,
      evolutionStrategy = config.tunerConfig.tunerEvolutionStrategy,
      continuousEvolutionMaxIterations =
        config.tunerConfig.tunerContinuousEvolutionMaxIterations,
      continuousEvolutionStoppingScore =
        config.tunerConfig.tunerContinuousEvolutionStoppingScore,
      continuousEvolutionParallelism =
        config.tunerConfig.tunerContinuousEvolutionParallelism,
      continuousEvolutionMutationAggressiveness =
        config.tunerConfig.tunerContinuousEvolutionMutationAggressiveness,
      continuousEvolutionGeneticMixing =
        config.tunerConfig.tunerContinuousEvolutionGeneticMixing,
      continuousEvolutionRollingImprovementCount =
        config.tunerConfig.tunerContinuousEvolutionRollingImprovingCount,
      dataReductionFactor = config.featureEngineeringConfig.dataReductionFactor,
      firstGenMode = config.tunerConfig.tunerInitialGenerationMode,
      firstGenPermutations =
        config.tunerConfig.tunerInitialGenerationPermutationCount,
      firstGenIndexMixingMode =
        config.tunerConfig.tunerInitialGenerationIndexMixingMode,
      firstGenArraySeed = config.tunerConfig.tunerInitialGenerationArraySeed,
      fieldsToIgnore = config.genericConfig.fieldsToIgnoreInVector,
      numericFillStat = config.featureEngineeringConfig.numericFillStat,
      characterFillStat = config.featureEngineeringConfig.characterFillStat,
      modelSelectionDistinctThreshold =
        config.featureEngineeringConfig.modelSelectionDistinctThreshold,
      dateTimeConversionType = config.genericConfig.dateTimeConversionType,
      modelType = config.predictionType,
      featureImportanceModelFamily = config.modelFamily,
      featureInteractionFlag = config.switchConfig.featureInteractionFlag,
      featureInteractionRetentionMode =
        config.featureEngineeringConfig.featureInteractionRetentionMode,
      featureInteractionContinuousDiscretizerBucketCount =
        config.featureEngineeringConfig.featureInteractionContinuousDiscretizerBucketCount,
      featureInteractionParallelism =
        config.featureEngineeringConfig.featureInteractionParallelism,
      featureInteractionTargetInteractionPercentage =
        config.featureEngineeringConfig.featureInteractionTargetInteractionPercentage
    )

  }

  /**
    * Method for generating a default configuration if no overrides are specified
    * @param modelFamily The model family (i.e. RandomForest) to generate a tuned model for
    * @param predictionType The type of prediction being made (either classifier or regressor)
    * @return MainConfig instance
    * @since 0.4.0
    * @author Ben Wilson, Databricks
    */
  def generateDefaultMainConfig(modelFamily: String,
                                predictionType: String): MainConfig = {
    val defaultInstanceConfig =
      generateDefaultConfig(modelFamily, predictionType)
    generateMainConfig(defaultInstanceConfig)
  }

  /**
    *
    * @param config
    * @return
    */
  def generatePrettyJsonInstanceConfig(config: InstanceConfig): String = {

    implicit val formats: Formats = Serialization.formats(hints = NoTypeHints)
    writePretty(config)
  }

  /**
    * User quality of life method for showing human readable config in pseudo-Map form
    * @param config Instance config object
    */
  def printFullConfig(config: InstanceConfig): Unit = {

    implicit val formats: Formats =
      Serialization.formats(hints = FullTypeHints(List(config.getClass)))
    println(
      writePretty(config)
        .replaceAll(": \\{", "\\{")
        .replaceAll(":", "->")
    )
  }

  /**
    * User quality of life method for showing human readable config in pseudo-Map form
    * @param config Instance config object
    */
  def printFullConfig(config: MainConfig): Unit = {

    implicit val formats: Formats =
      Serialization.formats(hints = FullTypeHints(List(config.getClass)))
    println(
      writePretty(config)
        .replaceAll(": \\{", "\\{")
        .replaceAll(":", "->")
    )
  }

  /**
    *
    * @param json
    * @return
    */
  def generateInstanceConfigFromJson(json: String): InstanceConfig = {
    implicit val formats: Formats = Serialization.formats(hints = NoTypeHints)
    read[InstanceConfig](json)
  }

  private def validateMapConfig(defaultMap: Map[String, Any],
                                submittedMap: Map[String, Any]): Unit = {

    val definedKeys = defaultMap.keys
    val submittedKeys = submittedMap.keys

    // Perform a quick-check

    val contained = submittedKeys.forall(definedKeys.toList.contains)
    if (!contained) {

      val invalidKeys = ListBuffer[String]()

      submittedKeys.map(
        x => if (!definedKeys.toList.contains(x)) invalidKeys += x
      )

      throw new IllegalArgumentException(
        s"Invalid map key(s) submitted for configuration generation. \nInvalid Keys: " +
          s"'${invalidKeys.mkString("','")}'. \n\tTo see a list of available keys, submit: \n\n\t\t" +
          s"ConfigurationGenerator.getConfigMapKeys \n\t\t\tor \n\t\tConfigurationGenerator.printConfigMapKeys \n\t\t\t " +
          s"to visualize in stdout.\n"
      )
    }

  }

  /**
    *
    * @param modelFamily
    * @param predictionType
    * @param config
    * @return
    */
  def generateConfigFromMap(modelFamily: String,
                            predictionType: String,
                            config: Map[String, Any]): InstanceConfig = {

    val defaultMap = defaultConfigMap(modelFamily, predictionType)

    // Validate the submitted keys to ensure that there are no invalid or mispelled entries
    validateMapConfig(defaultMap, config)

    lazy val genericConfigObject = new GenericConfigGenerator(predictionType)
      .setLabelCol(
        config.getOrElse("labelCol", defaultMap("labelCol")).toString
      )
      .setFeaturesCol(
        config.getOrElse("featuresCol", defaultMap("featuresCol")).toString
      )
      .setDateTimeConversionType(
        config
          .getOrElse(
            "dateTimeConversionType",
            defaultMap("dateTimeConversionType")
          )
          .toString
      )
      .setFieldsToIgnoreInVector(
        config
          .getOrElse(
            "fieldsToIgnoreInVector",
            defaultMap("fieldsToIgnoreInVector")
          )
          .asInstanceOf[Array[String]]
      )
      .setScoringMetric(
        config.getOrElse("scoringMetric", defaultMap("scoringMetric")).toString
      )
      .setScoringOptimizationStrategy(
        config
          .getOrElse(
            "scoringOptimizationStrategy",
            defaultMap("scoringOptimizationStrategy")
          )
          .toString
      )

    lazy val configObject = new ConfigurationGenerator(
      modelFamily,
      predictionType,
      genericConfigObject.getConfig
    ).setDataPrepParallelism(
        config
          .getOrElse("dataPrepParallelism", defaultMap("dataPrepParallelism"))
          .toString
          .toInt
      )
      .setNaFillFlag(
        config
          .getOrElse("naFillFlag", defaultMap("naFillFlag"))
          .toString
          .toBoolean
      )
      .setVarianceFilterFlag(
        config
          .getOrElse("varianceFilterFlag", defaultMap("varianceFilterFlag"))
          .toString
          .toBoolean
      )
      .setOutlierFilterFlag(
        config
          .getOrElse("outlierFilterFlag", defaultMap("outlierFilterFlag"))
          .toString
          .toBoolean
      )
      .setPearsonFilterFlag(
        config
          .getOrElse("pearsonFilterFlag", defaultMap("pearsonFilterFlag"))
          .toString
          .toBoolean
      )
      .setCovarianceFilterFlag(
        config
          .getOrElse("covarianceFilterFlag", defaultMap("covarianceFilterFlag"))
          .toString
          .toBoolean
      )
      .setOneHotEncodeFlag(
        config
          .getOrElse("oneHotEncodeFlag", defaultMap("oneHotEncodeFlag"))
          .toString
          .toBoolean
      )
      .setScalingFlag(
        config
          .getOrElse("scalingFlag", defaultMap("scalingFlag"))
          .toString
          .toBoolean
      )
      .setFeatureInteractionFlag(
        config
          .getOrElse(
            "featureInteractionFlag",
            defaultMap("featureInteractionFlag")
          )
          .toString
          .toBoolean
      )
      .setDataPrepCachingFlag(
        config
          .getOrElse("dataPrepCachingFlag", defaultMap("dataPrepCachingFlag"))
          .toString
          .toBoolean
      )
      .setAutoStoppingFlag(
        config
          .getOrElse("autoStoppingFlag", defaultMap("autoStoppingFlag"))
          .toString
          .toBoolean
      )
      .setPipelineDebugFlag(
        config
          .getOrElse("pipelineDebugFlag", defaultMap("pipelineDebugFlag"))
          .toString
          .toBoolean
      )
      .setFillConfigNumericFillStat(
        config
          .getOrElse(
            "fillConfigNumericFillStat",
            defaultMap("fillConfigNumericFillStat")
          )
          .toString
      )
      .setFillConfigCharacterFillStat(
        config
          .getOrElse(
            "fillConfigCharacterFillStat",
            defaultMap("fillConfigCharacterFillStat")
          )
          .toString
      )
      .setFillConfigModelSelectionDistinctThreshold(
        config
          .getOrElse(
            "fillConfigModelSelectionDistinctThreshold",
            defaultMap("fillConfigModelSelectionDistinctThreshold")
          )
          .toString
          .toInt
      )
      .setFillConfigCardinalitySwitch(
        config
          .getOrElse(
            "fillConfigCardinalitySwitch",
            defaultMap("fillConfigCardinalitySwitch")
          )
          .toString
          .toBoolean
      )
      .setFillConfigCardinalityType(
        config
          .getOrElse(
            "fillConfigCardinalityType",
            defaultMap("fillConfigCardinalityType")
          )
          .toString
      )
      .setFillConfigCardinalityPrecision(
        config
          .getOrElse(
            "fillConfigCardinalityPrecision",
            defaultMap("fillConfigCardinalityPrecision")
          )
          .toString
          .toDouble
      )
      .setFillConfigCardinalityCheckMode(
        config
          .getOrElse(
            "fillConfigCardinalityCheckMode",
            defaultMap("fillConfigCardinalityCheckMode")
          )
          .toString
      )
      .setFillConfigCardinalityLimit(
        config
          .getOrElse(
            "fillConfigCardinalityLimit",
            defaultMap("fillConfigCardinalityLimit")
          )
          .toString
          .toInt
      )
      .setFillConfigFilterPrecision(
        config
          .getOrElse(
            "fillConfigFilterPrecision",
            defaultMap("fillConfigFilterPrecision")
          )
          .toString
          .toDouble
      )
      .setFillConfigCategoricalNAFillMap(
        config
          .getOrElse(
            "fillConfigCategoricalNAFillMap",
            defaultMap("fillConfigCategoricalNAFillMap")
          )
          .asInstanceOf[Map[String, String]]
      )
      .setFillConfigNumericNAFillMap(
        config
          .getOrElse(
            "fillConfigNumericNAFillMap",
            defaultMap("fillConfigNumericNAFillMap")
          )
          .asInstanceOf[Map[String, AnyVal]]
      )
      .setFillConfigCharacterNABlanketFillValue(
        config
          .getOrElse(
            "fillConfigCharacterNABlanketFillValue",
            defaultMap("fillConfigCharacterNABlanketFillValue")
          )
          .toString
      )
      .setFillConfigNumericNABlanketFillValue(
        config
          .getOrElse(
            "fillConfigNumericNABlanketFillValue",
            defaultMap("fillConfigNumericNABlanketFillValue")
          )
          .toString
          .toDouble
      )
      .setFillConfigNAFillMode(
        config
          .getOrElse("fillConfigNAFillMode", defaultMap("fillConfigNAFillMode"))
          .toString
      )
      .setOutlierFilterBounds(
        config
          .getOrElse("outlierFilterBounds", defaultMap("outlierFilterBounds"))
          .toString
      )
      .setOutlierLowerFilterNTile(
        config
          .getOrElse(
            "outlierLowerFilterNTile",
            defaultMap("outlierLowerFilterNTile")
          )
          .toString
          .toDouble
      )
      .setOutlierUpperFilterNTile(
        config
          .getOrElse(
            "outlierUpperFilterNTile",
            defaultMap("outlierUpperFilterNTile")
          )
          .toString
          .toDouble
      )
      .setOutlierFilterPrecision(
        config
          .getOrElse(
            "outlierFilterPrecision",
            defaultMap("outlierFilterPrecision")
          )
          .toString
          .toDouble
      )
      .setOutlierContinuousDataThreshold(
        config
          .getOrElse(
            "outlierContinuousDataThreshold",
            defaultMap("outlierContinuousDataThreshold")
          )
          .toString
          .toInt
      )
      .setOutlierFieldsToIgnore(
        config
          .getOrElse(
            "outlierFieldsToIgnore",
            defaultMap("outlierFieldsToIgnore")
          )
          .asInstanceOf[Array[String]]
      )
      .setPearsonFilterStatistic(
        config
          .getOrElse(
            "pearsonFilterStatistic",
            defaultMap("pearsonFilterStatistic")
          )
          .toString
      )
      .setPearsonFilterDirection(
        config
          .getOrElse(
            "pearsonFilterDirection",
            defaultMap("pearsonFilterDirection")
          )
          .toString
      )
      .setPearsonFilterManualValue(
        config
          .getOrElse(
            "pearsonFilterManualValue",
            defaultMap("pearsonFilterManualValue")
          )
          .toString
          .toDouble
      )
      .setPearsonFilterMode(
        config
          .getOrElse("pearsonFilterMode", defaultMap("pearsonFilterMode"))
          .toString
      )
      .setPearsonAutoFilterNTile(
        config
          .getOrElse(
            "pearsonAutoFilterNTile",
            defaultMap("pearsonAutoFilterNTile")
          )
          .toString
          .toDouble
      )
      .setCovarianceCutoffLow(
        config
          .getOrElse("covarianceCutoffLow", defaultMap("covarianceCutoffLow"))
          .toString
          .toDouble
      )
      .setCovarianceCutoffHigh(
        config
          .getOrElse("covarianceCutoffHigh", defaultMap("covarianceCutoffHigh"))
          .toString
          .toDouble
      )
      .setScalingType(
        config.getOrElse("scalingType", defaultMap("scalingType")).toString
      )
      .setScalingMin(
        config
          .getOrElse("scalingMin", defaultMap("scalingMin"))
          .toString
          .toDouble
      )
      .setScalingMax(
        config
          .getOrElse("scalingMax", defaultMap("scalingMax"))
          .toString
          .toDouble
      )
      .setScalingStandardMeanFlag(
        config
          .getOrElse(
            "scalingStandardMeanFlag",
            defaultMap("scalingStandardMeanFlag")
          )
          .toString
          .toBoolean
      )
      .setScalingStdDevFlag(
        config
          .getOrElse("scalingStdDevFlag", defaultMap("scalingStdDevFlag"))
          .toString
          .toBoolean
      )
      .setScalingPNorm(
        config
          .getOrElse("scalingPNorm", defaultMap("scalingPNorm"))
          .toString
          .toDouble
      )
      .setFeatureInteractionRetentionMode(
        config
          .getOrElse(
            "featureInteractionRetentionMode",
            defaultMap("featureInteractionRetentionMode")
          )
          .toString
      )
      .setFeatureInteractionContinuousDiscretizerBucketCount(
        config
          .getOrElse(
            "featureInteractionContinuousDiscretizerBucketCount",
            defaultMap("featureInteractionContinuousDiscretizerBucketCount")
          )
          .toString
          .toInt
      )
      .setFeatureInteractionParallelism(
        config
          .getOrElse(
            "featureInteractionParallelism",
            defaultMap("featureInteractionParallelism")
          )
          .toString
          .toInt
      )
      .setFeatureInteractionTargetInteractionPercentage(
        config
          .getOrElse(
            "featureInteractionTargetInteractionPercentage",
            defaultMap("featureInteractionTargetInteractionPercentage")
          )
          .toString
          .toDouble
      )
      .setFeatureImportanceCutoffType(
        config
          .getOrElse(
            "featureImportanceCutoffType",
            defaultMap("featureImportanceCutoffType")
          )
          .toString
      )
      .setFeatureImportanceCutoffValue(
        config
          .getOrElse(
            "featureImportanceCutoffValue",
            defaultMap("featureImportanceCutoffValue")
          )
          .toString
          .toDouble
      )
      .setDataReductionFactor(
        config
          .getOrElse("dataReductionFactor", defaultMap("dataReductionFactor"))
          .toString
          .toDouble
      )
      .setStringBoundaries(
        config
          .getOrElse("stringBoundaries", defaultMap("stringBoundaries"))
          .asInstanceOf[Map[String, List[String]]]
      )
      .setNumericBoundaries(
        config
          .getOrElse("numericBoundaries", defaultMap("numericBoundaries"))
          .asInstanceOf[Map[String, (Double, Double)]]
      )
      .setTunerAutoStoppingScore(
        config
          .getOrElse(
            "tunerAutoStoppingScore",
            defaultMap("tunerAutoStoppingScore")
          )
          .toString
          .toDouble
      )
      .setTunerParallelism(
        config
          .getOrElse("tunerParallelism", defaultMap("tunerParallelism"))
          .toString
          .toInt
      )
      .setTunerKFold(
        config.getOrElse("tunerKFold", defaultMap("tunerKFold")).toString.toInt
      )
      .setTunerTrainPortion(
        config
          .getOrElse("tunerTrainPortion", defaultMap("tunerTrainPortion"))
          .toString
          .toDouble
      )
      .setTunerTrainSplitMethod(
        config
          .getOrElse(
            "tunerTrainSplitMethod",
            defaultMap("tunerTrainSplitMethod")
          )
          .toString
      )
      .setTunerKSampleSyntheticCol(
        config
          .getOrElse(
            "tunerKSampleSyntheticCol",
            defaultMap("tunerKSampleSyntheticCol")
          )
          .toString
      )
      .setTunerKSampleKGroups(
        config
          .getOrElse("tunerKSampleKGroups", defaultMap("tunerKSampleKGroups"))
          .toString
          .toInt
      )
      .setTunerKSampleKMeansMaxIter(
        config
          .getOrElse(
            "tunerKSampleKMeansMaxIter",
            defaultMap("tunerKSampleKMeansMaxIter")
          )
          .toString
          .toInt
      )
      .setTunerKSampleKMeansTolerance(
        config
          .getOrElse(
            "tunerKSampleKMeansTolerance",
            defaultMap("tunerKSampleKMeansTolerance")
          )
          .toString
          .toDouble
      )
      .setTunerKSampleKMeansDistanceMeasurement(
        config
          .getOrElse(
            "tunerKSampleKMeansDistanceMeasurement",
            defaultMap("tunerKSampleKMeansDistanceMeasurement")
          )
          .toString
      )
      .setTunerKSampleKMeansSeed(
        config
          .getOrElse(
            "tunerKSampleKMeansSeed",
            defaultMap("tunerKSampleKMeansSeed")
          )
          .toString
          .toLong
      )
      .setTunerKSampleKMeansPredictionCol(
        config
          .getOrElse(
            "tunerKSampleKMeansPredictionCol",
            defaultMap("tunerKSampleKMeansPredictionCol")
          )
          .toString
      )
      .setTunerKSampleLSHHashTables(
        config
          .getOrElse(
            "tunerKSampleLSHHashTables",
            defaultMap("tunerKSampleLSHHashTables")
          )
          .toString
          .toInt
      )
      .setTunerKSampleLSHSeed(
        config
          .getOrElse("tunerKSampleLSHSeed", defaultMap("tunerKSampleLSHSeed"))
          .toString
          .toLong
      )
      .setTunerKSampleLSHOutputCol(
        config
          .getOrElse(
            "tunerKSampleLSHOutputCol",
            defaultMap("tunerKSampleLSHOutputCol")
          )
          .toString
      )
      .setTunerKSampleQuorumCount(
        config
          .getOrElse(
            "tunerKSampleQuorumCount",
            defaultMap("tunerKSampleQuorumCount")
          )
          .toString
          .toInt
      )
      .setTunerKSampleMinimumVectorCountToMutate(
        config
          .getOrElse(
            "tunerKSampleMinimumVectorCountToMutate",
            defaultMap("tunerKSampleMinimumVectorCountToMutate")
          )
          .toString
          .toInt
      )
      .setTunerKSampleVectorMutationMethod(
        config
          .getOrElse(
            "tunerKSampleVectorMutationMethod",
            defaultMap("tunerKSampleVectorMutationMethod")
          )
          .toString
      )
      .setTunerKSampleMutationMode(
        config
          .getOrElse(
            "tunerKSampleMutationMode",
            defaultMap("tunerKSampleMutationMode")
          )
          .toString
      )
      .setTunerKSampleMutationValue(
        config
          .getOrElse(
            "tunerKSampleMutationValue",
            defaultMap("tunerKSampleMutationValue")
          )
          .toString
          .toDouble
      )
      .setTunerKSampleLabelBalanceMode(
        config
          .getOrElse(
            "tunerKSampleLabelBalanceMode",
            defaultMap("tunerKSampleLabelBalanceMode")
          )
          .toString
      )
      .setTunerKSampleCardinalityThreshold(
        config
          .getOrElse(
            "tunerKSampleCardinalityThreshold",
            defaultMap("tunerKSampleCardinalityThreshold")
          )
          .toString
          .toInt
      )
      .setTunerKSampleNumericRatio(
        config
          .getOrElse(
            "tunerKSampleNumericRatio",
            defaultMap("tunerKSampleNumericRatio")
          )
          .toString
          .toDouble
      )
      .setTunerKSampleNumericTarget(
        config
          .getOrElse(
            "tunerKSampleNumericTarget",
            defaultMap("tunerKSampleNumericTarget")
          )
          .toString
          .toInt
      )
      .setTunerOutputDfRepartitionScaleFactor(
        config
          .getOrElse(
            "tunerOutputDfRepartitionScaleFactor",
            defaultMap("tunerOutputDfRepartitionScaleFactor")
          )
          .toString
          .toInt
      )
      .setTunerTrainSplitChronologicalColumn(
        config
          .getOrElse(
            "tunerTrainSplitChronologicalColumn",
            defaultMap("tunerTrainSplitChronologicalColumn")
          )
          .toString
      )
      .setTunerTrainSplitChronologicalRandomPercentage(
        config
          .getOrElse(
            "tunerTrainSplitChronologicalRandomPercentage",
            defaultMap("tunerTrainSplitChronologicalRandomPercentage")
          )
          .toString
          .toDouble
      )
      .setTunerSeed(
        config.getOrElse("tunerSeed", defaultMap("tunerSeed")).toString.toLong
      )
      .setTunerFirstGenerationGenePool(
        config
          .getOrElse(
            "tunerFirstGenerationGenePool",
            defaultMap("tunerFirstGenerationGenePool")
          )
          .toString
          .toInt
      )
      .setTunerNumberOfGenerations(
        config
          .getOrElse(
            "tunerNumberOfGenerations",
            defaultMap("tunerNumberOfGenerations")
          )
          .toString
          .toInt
      )
      .setTunerNumberOfParentsToRetain(
        config
          .getOrElse(
            "tunerNumberOfParentsToRetain",
            defaultMap("tunerNumberOfParentsToRetain")
          )
          .toString
          .toInt
      )
      .setTunerNumberOfMutationsPerGeneration(
        config
          .getOrElse(
            "tunerNumberOfMutationsPerGeneration",
            defaultMap("tunerNumberOfMutationsPerGeneration")
          )
          .toString
          .toInt
      )
      .setTunerGeneticMixing(
        config
          .getOrElse("tunerGeneticMixing", defaultMap("tunerGeneticMixing"))
          .toString
          .toDouble
      )
      .setTunerGenerationalMutationStrategy(
        config
          .getOrElse(
            "tunerGenerationalMutationStrategy",
            defaultMap("tunerGenerationalMutationStrategy")
          )
          .toString
      )
      .setTunerFixedMutationValue(
        config
          .getOrElse(
            "tunerFixedMutationValue",
            defaultMap("tunerFixedMutationValue")
          )
          .toString
          .toInt
      )
      .setTunerMutationMagnitudeMode(
        config
          .getOrElse(
            "tunerMutationMagnitudeMode",
            defaultMap("tunerMutationMagnitudeMode")
          )
          .toString
      )
      .setTunerEvolutionStrategy(
        config
          .getOrElse(
            "tunerEvolutionStrategy",
            defaultMap("tunerEvolutionStrategy")
          )
          .toString
      )
      .setTunerGeneticMBORegressorType(
        config
          .getOrElse(
            "tunerGeneticMBORegressorType",
            defaultMap("tunerGeneticMBORegressorType")
          )
          .toString
      )
      .setTunerGeneticMBOCandidateFactor(
        config
          .getOrElse(
            "tunerGeneticMBOCandidateFactor",
            defaultMap("tunerGeneticMBOCandidateFactor")
          )
          .toString
          .toInt
      )
      .setTunerContinuousEvolutionImprovementThreshold(
        config
          .getOrElse(
            "tunerContinuousEvolutionImprovementThreshold",
            defaultMap("tunerContinuousEvolutionImprovementThreshold")
          )
          .toString
          .toInt
      )
      .setTunerContinuousEvolutionMaxIterations(
        config
          .getOrElse(
            "tunerContinuousEvolutionMaxIterations",
            defaultMap("tunerContinuousEvolutionMaxIterations")
          )
          .toString
          .toInt
      )
      .setTunerContinuousEvolutionStoppingScore(
        config
          .getOrElse(
            "tunerContinuousEvolutionStoppingScore",
            defaultMap("tunerContinuousEvolutionStoppingScore")
          )
          .toString
          .toDouble
      )
      .setTunerContinuousEvolutionParallelism(
        config
          .getOrElse(
            "tunerContinuousEvolutionParallelism",
            defaultMap("tunerContinuousEvolutionParallelism")
          )
          .toString
          .toInt
      )
      .setTunerContinuousEvolutionMutationAggressiveness(
        config
          .getOrElse(
            "tunerContinuousEvolutionMutationAggressiveness",
            defaultMap("tunerContinuousEvolutionMutationAggressiveness")
          )
          .toString
          .toInt
      )
      .setTunerContinuousEvolutionGeneticMixing(
        config
          .getOrElse(
            "tunerContinuousEvolutionGeneticMixing",
            defaultMap("tunerContinuousEvolutionGeneticMixing")
          )
          .toString
          .toDouble
      )
      .setTunerContinuousEvolutionRollingImprovementCount(
        config
          .getOrElse(
            "tunerContinuousEvolutionRollingImprovementCount",
            defaultMap("tunerContinuousEvolutionRollingImprovementCount")
          )
          .toString
          .toInt
      )
      .setTunerModelSeed(
        config
          .getOrElse("tunerModelSeed", defaultMap("tunerModelSeed"))
          .asInstanceOf[Map[String, Any]]
      )
      .setTunerHyperSpaceInferenceFlag(
        config
          .getOrElse(
            "tunerHyperSpaceInferenceFlag",
            defaultMap("tunerHyperSpaceInferenceFlag")
          )
          .toString
          .toBoolean
      )
      .setTunerHyperSpaceInferenceCount(
        config
          .getOrElse(
            "tunerHyperSpaceInferenceCount",
            defaultMap("tunerHyperSpaceInferenceCount")
          )
          .toString
          .toInt
      )
      .setTunerHyperSpaceModelCount(
        config
          .getOrElse(
            "tunerHyperSpaceModelCount",
            defaultMap("tunerHyperSpaceModelCount")
          )
          .toString
          .toInt
      )
      .setTunerHyperSpaceModelType(
        config
          .getOrElse(
            "tunerHyperSpaceModelType",
            defaultMap("tunerHyperSpaceModelType")
          )
          .toString
      )
      .setTunerInitialGenerationMode(
        config
          .getOrElse(
            "tunerInitialGenerationMode",
            defaultMap("tunerInitialGenerationMode")
          )
          .toString
      )
      .setTunerInitialGenerationPermutationCount(
        config
          .getOrElse(
            "tunerInitialGenerationPermutationCount",
            defaultMap("tunerInitialGenerationPermutationCount")
          )
          .toString
          .toInt
      )
      .setTunerInitialGenerationIndexMixingMode(
        config
          .getOrElse(
            "tunerInitialGenerationIndexMixingMode",
            defaultMap("tunerInitialGenerationIndexMixingMode")
          )
          .toString
      )
      .setTunerInitialGenerationArraySeed(
        config
          .getOrElse(
            "tunerInitialGenerationArraySeed",
            defaultMap("tunerInitialGenerationArraySeed")
          )
          .toString
          .toLong
      )
      .setMlFlowLoggingFlag(
        config
          .getOrElse("mlFlowLoggingFlag", defaultMap("mlFlowLoggingFlag"))
          .toString
          .toBoolean
      )
      .setMlFlowLogArtifactsFlag(
        config
          .getOrElse(
            "mlFlowLogArtifactsFlag",
            defaultMap("mlFlowLogArtifactsFlag")
          )
          .toString
          .toBoolean
      )
      .setMlFlowTrackingURI(
        config
          .getOrElse("mlFlowTrackingURI", defaultMap("mlFlowTrackingURI"))
          .toString
      )
      .setMlFlowExperimentName(
        config
          .getOrElse("mlFlowExperimentName", defaultMap("mlFlowExperimentName"))
          .toString
      )
      .setMlFlowAPIToken(
        config
          .getOrElse("mlFlowAPIToken", defaultMap("mlFlowAPIToken"))
          .toString
      )
      .setMlFlowModelSaveDirectory(
        config
          .getOrElse(
            "mlFlowModelSaveDirectory",
            defaultMap("mlFlowModelSaveDirectory")
          )
          .toString
      )
      .setMlFlowLoggingMode(
        config
          .getOrElse("mlFlowLoggingMode", defaultMap("mlFlowLoggingMode"))
          .toString
      )
      .setMlFlowBestSuffix(
        config
          .getOrElse("mlFlowBestSuffix", defaultMap("mlFlowBestSuffix"))
          .toString
      )
      .setInferenceConfigSaveLocation(
        config
          .getOrElse(
            "inferenceConfigSaveLocation",
            defaultMap("inferenceConfigSaveLocation")
          )
          .toString
      )
      .setMlFlowCustomRunTags(
        config
          .getOrElse("mlFlowCustomRunTags", defaultMap("mlFlowCustomRunTags"))
          .asInstanceOf[Map[String, AnyVal]]
      )

    configObject.getInstanceConfig

  }

  def getDefaultConfigMap(modelFamily: String,
                          predictionType: String): Map[String, Any] =
    defaultConfigMap(modelFamily, predictionType)

  def getConfigMapKeys: Iterable[String] =
    defaultConfigMap("randomForest", "classifier").keys

  def printConfigMapKeys(): Unit = { getConfigMapKeys.foreach(println(_)) }

}
