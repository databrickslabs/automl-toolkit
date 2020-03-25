package com.databricks.labs.automl.feature

import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.utilities.ValidationUtilities
import com.databricks.labs.automl.{
  AbstractUnitSpec,
  AutomationRunner,
  AutomationUnitTestsUtil
}

class FeatureInteractionKSampleIntegrationTest extends AbstractUnitSpec {

  it should "Perform Data Prep Correctly with FeatureInteraction and kSampling both on" in {

    val EXPECTED_FIELDS = Array(
      "age_trimmed_si",
      "workclass_trimmed_si",
      "education_trimmed_si",
      "education-num_trimmed_si",
      "marital-status_trimmed_si",
      "occupation_trimmed_si",
      "relationship_trimmed_si",
      "race_trimmed_si",
      "sex_trimmed_si",
      "capital-gain_trimmed_si",
      "capital-loss_trimmed_si",
      "hours-per-week_trimmed_si",
      "native-country_trimmed_si",
      "i_age_trimmed_si_workclass_trimmed_si_si",
      "i_relationship_trimmed_si_sex_trimmed_si_si",
      "i_education-num_trimmed_si_sex_trimmed_si_si",
      "i_education-num_trimmed_si_race_trimmed_si_si",
      "i_occupation_trimmed_si_capital-gain_trimmed_si_si",
      "i_workclass_trimmed_si_capital-gain_trimmed_si_si",
      "i_education_trimmed_si_capital-gain_trimmed_si_si",
      "i_marital-status_trimmed_si_hours-per-week_trimmed_si_si",
      "i_age_trimmed_si_occupation_trimmed_si_si",
      "i_education-num_trimmed_si_occupation_trimmed_si_si",
      "i_age_trimmed_si_capital-loss_trimmed_si_si",
      "i_education-num_trimmed_si_capital-loss_trimmed_si_si",
      "i_age_trimmed_si_education_trimmed_si_si",
      "i_relationship_trimmed_si_capital-gain_trimmed_si_si",
      "i_education-num_trimmed_si_capital-gain_trimmed_si_si",
      "i_occupation_trimmed_si_capital-loss_trimmed_si_si",
      "i_workclass_trimmed_si_capital-loss_trimmed_si_si",
      "i_education_trimmed_si_capital-loss_trimmed_si_si",
      "i_marital-status_trimmed_si_native-country_trimmed_si_si",
      "i_age_trimmed_si_relationship_trimmed_si_si",
      "i_education-num_trimmed_si_relationship_trimmed_si_si",
      "i_age_trimmed_si_hours-per-week_trimmed_si_si",
      "i_education-num_trimmed_si_hours-per-week_trimmed_si_si",
      "i_age_trimmed_si_education-num_trimmed_si_si",
      "i_relationship_trimmed_si_capital-loss_trimmed_si_si",
      "i_marital-status_trimmed_si_relationship_trimmed_si_si",
      "i_sex_trimmed_si_capital-gain_trimmed_si_si",
      "i_occupation_trimmed_si_hours-per-week_trimmed_si_si",
      "i_workclass_trimmed_si_hours-per-week_trimmed_si_si",
      "i_education_trimmed_si_hours-per-week_trimmed_si_si",
      "i_occupation_trimmed_si_relationship_trimmed_si_si",
      "i_age_trimmed_si_race_trimmed_si_si",
      "i_marital-status_trimmed_si_sex_trimmed_si_si",
      "i_age_trimmed_si_native-country_trimmed_si_si",
      "i_education-num_trimmed_si_native-country_trimmed_si_si",
      "i_age_trimmed_si_marital-status_trimmed_si_si",
      "i_relationship_trimmed_si_hours-per-week_trimmed_si_si",
      "i_marital-status_trimmed_si_race_trimmed_si_si",
      "i_sex_trimmed_si_capital-loss_trimmed_si_si",
      "i_occupation_trimmed_si_native-country_trimmed_si_si",
      "i_workclass_trimmed_si_native-country_trimmed_si_si",
      "i_education_trimmed_si_native-country_trimmed_si_si",
      "i_occupation_trimmed_si_race_trimmed_si_si",
      "i_age_trimmed_si_sex_trimmed_si_si",
      "i_marital-status_trimmed_si_capital-gain_trimmed_si_si",
      "i_workclass_trimmed_si_education_trimmed_si_si",
      "i_marital-status_trimmed_si_occupation_trimmed_si_si",
      "i_relationship_trimmed_si_native-country_trimmed_si_si",
      "i_sex_trimmed_si_hours-per-week_trimmed_si_si",
      "i_relationship_trimmed_si_race_trimmed_si_si",
      "i_education_trimmed_si_education-num_trimmed_si_si",
      "i_education-num_trimmed_si_marital-status_trimmed_si_si",
      "i_occupation_trimmed_si_sex_trimmed_si_si",
      "i_marital-status_trimmed_si_capital-loss_trimmed_si_si",
      "i_age_trimmed_si_capital-gain_trimmed_si_si",
      "i_workclass_trimmed_si_education-num_trimmed_si_si",
      "i_race_trimmed_si_sex_trimmed_si_si",
      "i_sex_trimmed_si_native-country_trimmed_si_si",
      "i_workclass_trimmed_si_marital-status_trimmed_si_si",
      "i_education_trimmed_si_marital-status_trimmed_si_si",
      "i_workclass_trimmed_si_relationship_trimmed_si_si",
      "i_workclass_trimmed_si_occupation_trimmed_si_si",
      "i_capital-gain_trimmed_si_hours-per-week_trimmed_si_si",
      "i_workclass_trimmed_si_race_trimmed_si_si",
      "i_capital-loss_trimmed_si_hours-per-week_trimmed_si_si",
      "i_race_trimmed_si_capital-gain_trimmed_si_si",
      "i_capital-gain_trimmed_si_capital-loss_trimmed_si_si",
      "i_education_trimmed_si_occupation_trimmed_si_si",
      "i_workclass_trimmed_si_sex_trimmed_si_si",
      "i_capital-loss_trimmed_si_native-country_trimmed_si_si",
      "i_capital-gain_trimmed_si_native-country_trimmed_si_si",
      "i_hours-per-week_trimmed_si_native-country_trimmed_si_si",
      "i_race_trimmed_si_capital-loss_trimmed_si_si",
      "i_education_trimmed_si_relationship_trimmed_si_si",
      "i_race_trimmed_si_hours-per-week_trimmed_si_si",
      "i_education_trimmed_si_race_trimmed_si_si",
      "i_race_trimmed_si_native-country_trimmed_si_si",
      "i_education_trimmed_si_sex_trimmed_si_si",
      "features",
      "label",
      "synthetic_ksample"
    )

    val testData = AutomationUnitTestsUtil.getAdultDf()

    val runConfig = Map(
      "labelCol" -> "label",
      "tunerKFold" -> 3,
      "tunerTrainSplitMethod" -> "kSample",
      "featureInteractionFlag" -> true,
      "featureInteractionRetentionMode" -> "all",
      "tunerNumberOfGenerations" -> 3,
      "tunerInitialGenerationMode" -> "permutations",
      "mlFlowLoggingFlag" -> false
    )

    val rfConfig = ConfigurationGenerator.generateConfigFromMap(
      "RandomForest",
      "classifier",
      runConfig
    )

    val prep = new AutomationRunner(testData)
      .setMainConfig(ConfigurationGenerator.generateMainConfig(rfConfig))
      .prepData()

    println(prep.data.schema.names.mkString(", "))

    assert(
      prep.modelType == "classifier",
      s"model detection type incorrect: ${prep.modelType}, should have been classifier"
    )
    assert(!prep.data.isEmpty, "prepared Dataframe should not be empty.")
    ValidationUtilities.fieldCreationAssertion(
      EXPECTED_FIELDS,
      prep.data.schema.names
    )
    ValidationUtilities.fieldCreationAssertion(EXPECTED_FIELDS, prep.fields)

  }

}
