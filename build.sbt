

name := "automl-toolkit"

organization := "com.databricks.labs"

version := "0.8.1"

scalaVersion := "2.12.10"
scalacOptions ++= Seq("-Xmax-classfile-name", "78")

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.1" % Provided
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.1" % Provided
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.1" % Provided
libraryDependencies += "org.mlflow" % "mlflow-client" % "1.11.0" % Provided
libraryDependencies += "org.json4s" %% "json4s-jackson" % "3.6.6" % Provided
libraryDependencies += "ml.dmlc" %% "xgboost4j" % "1.0.0" % Provided
libraryDependencies += "ml.dmlc" %% "xgboost4j-spark" % "1.0.0" % Provided
libraryDependencies += "org.joda"% "joda-convert" % "1.7" % Test
libraryDependencies += "joda-time" % "joda-time" % "2.10.5" % Test
libraryDependencies += "junit" % "junit" % "4.13.1" % Test
libraryDependencies += "org.scalatest" % "scalatest_2.12" % "3.0.8"
libraryDependencies += "com.databricks" %% "dbutils-api" % "0.0.5" % Provided
libraryDependencies += "ml.combust.mleap" %% "mleap-runtime" % "0.16.0"
libraryDependencies += "ml.combust.mleap" %% "mleap-spark" % "0.16.0"
//libraryDependencies += "com.microsoft.ml.spark" % "mmlspark_2.11" % "0.18.1"
//libraryDependencies += "org.vegas-viz" %% "vegas" % "0.3.11"

lazy val commonSettings = Seq(
  version := "0.8.1",
  organization := "com.databricks.labs",
  scalaVersion := "2.12.10"
)

assemblyShadeRules in assembly := Seq(
  ShadeRule.rename("org.json4s.**" -> "shadeio.@1").inAll
)

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x                             => MergeStrategy.first
}

assemblyExcludedJars in assembly := {
  val cp = (fullClasspath in assembly).value
  cp filter { f =>
    f.data.getName.contains("spark-core") ||
    f.data.getName.contains("spark-mllib") ||
    f.data.getName.contains("spark-sql") ||
    f.data.getName.contains("com.databricks.backend") ||
    f.data.getName.contains("com.microsoft.ml.spark") ||
    f.data.getName.contains("com.databricks.dbutils-api_2.12")
  }
}

unmanagedSources / excludeFilter := HiddenFileFilter || "*GbmTunerDelegator.scala" || "*LightGBMTuner.scala" || "*OneDimVisualizations.scala"
Test / unmanagedSources / excludeFilter := HiddenFileFilter || "*OneDimVisualizationsTest.scala"
