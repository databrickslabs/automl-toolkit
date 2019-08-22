name := "AutomatedML"

organization := "com.databricks"

version := "0.5.2"

scalaVersion := "2.11.12"
scalacOptions ++= Seq("-Xmax-classfile-name","78")

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.0"
libraryDependencies += "org.mlflow" % "mlflow-client" % "0.9.1"
libraryDependencies += "org.json4s" %% "json4s-jackson" % "3.5.3"
libraryDependencies += "ml.dmlc" % "xgboost4j" % "0.90"
libraryDependencies += "ml.dmlc" % "xgboost4j-spark" % "0.90"
libraryDependencies += "junit" % "junit" % "4.8.1" % "test"
libraryDependencies += "org.scalatest" % "scalatest_2.11" % "3.0.6" % "test"

lazy val commonSettings = Seq(
  version := "0.5.1",
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
    f.data.getName.contains("spark-sql")
  }
}
