name := "AutomatedML"

organization := "com.databricks"

version := "0.4.2"

scalaVersion := "2.11.12"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.0"
libraryDependencies += "org.mlflow" % "mlflow-client" % "0.8.1"
libraryDependencies += "org.json4s" %% "json4s-native" % "3.6.4"


lazy val commonSettings = Seq(
  version := "0.4.2",
  organization := "com.databricks",
  scalaVersion := "2.11.12"
)

assemblyShadeRules in assembly := Seq(
  ShadeRule.rename("org.json4s.**" -> "shadeio.@1").inAll
)

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

assemblyExcludedJars in assembly := {
  val cp = (fullClasspath in assembly).value
  cp filter { f =>
    f.data.getName.contains("spark-core") ||
    f.data.getName.contains("spark-mllib") ||
    f.data.getName.contains("spark-sql")
  }
}