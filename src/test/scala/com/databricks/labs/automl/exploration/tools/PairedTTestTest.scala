package com.databricks.labs.automl.exploration.tools

import com.databricks.labs.automl.AbstractUnitSpec

class PairedTTestTest extends AbstractUnitSpec {

  it should "correctly classify ANOVA non-equivalency with identical distribution types" in {

    val expectedCorrelation = 0.9867499999999999
    val expectedPearsons = 0.5353263726770642
    val expectedSpearmans = 0.49848254581765294
    val expectedKendalls = 0.44946657497549475

    val expectedTTestPValue = 8.487543144559928E-4
    val expectedTTestStat = -4.763582449331357
    val expectedTTestEquivalencyJudgement = 'N'
    val expectedKSDStat = 1.0
    val expectedKSPValue = 0.0
    val expectedKSJudgement = 'Y'

    val left = Seq(0.1, 0.02, 0.399, 0.4, 0.566, 0.6, 0.7, 0.8, 0.9, 1.0)
    val right = Seq(2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 11.0, 18.0, 4.0)

    val test = PairedTesting.evaluate(left, right, 0.05)

    assert(
      (test.tTestData.tTestPValue == expectedTTestPValue) && (test.tTestData.tStat == expectedTTestStat)
        && test.tTestData.tTestSignificance &&
        (test.tTestData.equivalencyJudgement == expectedTTestEquivalencyJudgement) &&
        (test.kolmogorovSmirnovData.ksTestDStatistic == expectedKSDStat) &&
        (test.kolmogorovSmirnovData.ksTestPvalue == expectedKSPValue) &&
        (test.kolmogorovSmirnovData.ksTestEquivalency == expectedKSJudgement) &&
        (test.correlationTestData.covariance == expectedCorrelation) &&
        (test.correlationTestData.pearsonCoefficient == expectedPearsons) &&
        (test.correlationTestData.spearmanCoefficient == expectedSpearmans) &&
        (test.correlationTestData.kendallsTauCoefficient == expectedKendalls),
      "values are incorrect"
    )

  }

  it should "correctly classify ANOVA equivalency with similar distribution types" in {

    val expectedCorrelation = -45280.167386206616
    val expectedPearsons = -0.47399155389027436
    val expectedSpearmans = 0.49090909090909085
    val expectedKendalls = 0.6
    val expectedTTestPValue = 0.3403272377813734
    val expectedTTestStat = -1.0012295762272705
    val expectedTTestEquivalencyJudgement = 'Y'
    val expectedKSDStat = 0.7272727272727273
    val expectedKSPValue = 6.549178375803762E-4
    val expectedKSJudgement = 'Y'

    val left =
      Seq(0.1, 0.02, 0.399, 0.4, 0.566, 0.6, 0.7, 0.8, 0.9, 1.0, 0.00001)
    val right =
      Seq(0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0, 1000.0, 1000000.0)

    val test = PairedTesting.evaluate(left, right, 0.05)

    assert(
      (test.tTestData.tTestPValue == expectedTTestPValue) && (test.tTestData.tStat == expectedTTestStat)
        && !test.tTestData.tTestSignificance &&
        (test.tTestData.equivalencyJudgement == expectedTTestEquivalencyJudgement) &&
        (test.kolmogorovSmirnovData.ksTestDStatistic == expectedKSDStat) &&
        (test.kolmogorovSmirnovData.ksTestPvalue == expectedKSPValue) &&
        (test.kolmogorovSmirnovData.ksTestEquivalency == expectedKSJudgement) &&
        (test.correlationTestData.covariance == expectedCorrelation) &&
        (test.correlationTestData.pearsonCoefficient == expectedPearsons) &&
        (test.correlationTestData.spearmanCoefficient == expectedSpearmans) &&
        (test.correlationTestData.kendallsTauCoefficient == expectedKendalls),
      "values are incorrect"
    )

  }

}
