name := "AutomatedML"

organization := "com.databricks"

version := "0.6.2"

scalaVersion := "2.11.12"
scalacOptions ++= Seq("-Xmax-classfile-name", "78")

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.0"
libraryDependencies += "org.mlflow" % "mlflow-client" % "1.3.0"
libraryDependencies += "org.json4s" %% "json4s-jackson" % "3.5.3"
libraryDependencies += "ml.dmlc" % "xgboost4j" % "0.90"
libraryDependencies += "ml.dmlc" % "xgboost4j-spark" % "0.90"
libraryDependencies += "junit" % "junit" % "4.8.1" % "test"
libraryDependencies += "org.scalatest" % "scalatest_2.11" % "3.0.6"
libraryDependencies += "com.databricks" % "dbutils-api_2.11" % "0.0.3"
libraryDependencies += "ml.combust.mleap" %% "mleap-runtime" % "0.14.0"
libraryDependencies += "ml.combust.mleap" %% "mleap-spark" % "0.14.0"
libraryDependencies += "com.microsoft.ml.spark" %% "mmlspark" % "0.18.1"

lazy val commonSettings = Seq(
  version := "0.6.2",
  organization := "com.databricks",
  scalaVersion := "2.11.12"
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
    f.data.getName.contains("com.microsoft.ml.spark")
  }
}
